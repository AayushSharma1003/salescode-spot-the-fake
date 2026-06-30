"""
Train the real-vs-screen detector and pickle the model.
Reads features from dataset/features.csv (built by build_dataset.ipynb),
fits SVM-RBF with calibrated probabilities, and saves {model, features}.
Run: python train.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# sklearn 1.9+ flags SVC(probability=True) as deprecated for 1.11. The
# replacement (CalibratedClassifierCV) has slightly noisier calibration on
# small data, which shifts our held-out numbers. Stay with what we validated.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# The 7 features that survived ablation. Order matters: predict.py rebuilds
# the vector in this order, so changing it requires retraining.
FEATS = [
    "moire_max", "moire_mean", "moire_p90",
    "hf_ratio", "banding",
    "residual_std", "residual_kurt",
]

CSV = Path(__file__).parent / "dataset" / "features.csv"
OUT = Path(__file__).parent / "model.pkl"


def make_model():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=2, probability=True, random_state=0),
    )


def loso_eval(df):
    """Leave-one-screen-out: train on two screens, test on the held-out third
    plus a fixed half-sample of reals. This is the honest cross-device number."""
    rng = np.random.RandomState(0)
    real_idx = rng.permutation(df[df.label == 0].index.values)
    held_reals = set(real_idx[: len(real_idx) // 2])

    # Predict via predict_proba >= 0.5 (not .predict()) so the eval matches
    # what predict.py returns to the reviewer -- SVC.predict uses the raw
    # decision function, which can disagree with the calibrated probability.
    print("\nLeave-one-screen-out (held-out screen + held-out reals):")
    for S in ["hp", "mac", "op"]:
        te = (df.screen_type == S) | (df.index.isin(held_reals) & (df.label == 0))
        m = make_model().fit(df.loc[~te, FEATS], df.loc[~te, "label"])
        proba = m.predict_proba(df.loc[te, FEATS])[:, 1]
        pred = (proba >= 0.5).astype(int)
        truth = df.loc[te, "label"].values
        screen_mask = (df.loc[te, "screen_type"] == S).values
        acc = (pred == truth).mean()
        rec = pred[screen_mask].mean()
        print(f"  held out {S:>3}: acc {acc:.1%} | caught {rec:.0%} of unseen {S} recaptures")


def main():
    df = pd.read_csv(CSV)
    X, y = df[FEATS].values, df["label"].values
    print(f"Loaded {len(df)} images "
          f"({(y == 0).sum()} real, {(y == 1).sum()} screen)")
    print(f"Features ({len(FEATS)}): {FEATS}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(make_model(), X, y, cv=cv, scoring="accuracy")
    print(f"\n5-fold CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    loso_eval(df)

    # Fit on all data for the shipped model.
    final = make_model().fit(X, y)
    bundle = {"model": final, "features": FEATS}
    with open(OUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
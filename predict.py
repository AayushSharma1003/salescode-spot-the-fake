"""
Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import sys
import pickle
import warnings
from pathlib import Path

from features import extract_features

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

_BUNDLE_PATH = Path(__file__).parent / "model.pkl"
_bundle = None


def _load():
    # Lazy + cached so the pickle is read once per process.
    global _bundle
    if _bundle is None:
        with open(_BUNDLE_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def predict(image_path: str) -> float:
    b = _load()
    feats = extract_features(image_path)
    x = [[feats[k] for k in b["features"]]]
    return float(b["model"].predict_proba(x)[0, 1])


if __name__ == "__main__":
    print(predict(sys.argv[1]))
# Spot the Fake Photo — Note

**Live demo:** https://salescode-spot-the-fake.onrender.com — upload a photo or use your phone camera.

## Approach

The brief says *"small, fast, cheap, honest."* That ruled out a CNN for me: on 100 training images, a deep model would overfit, the inference cost climbs, and the rubric specifically rewards interpretability. The physics of the problem also points the other way — screen recaptures leave low-level frequency artefacts (LCD pixel grid moiré, reshoot blur, sensor noise differences) that hand-crafted features capture directly without 224×224 downsampling killing them.

So: 7 features per image → `StandardScaler` → `SVC(rbf, C=2, probability=True)`.

## Features

| Feature | What it measures | Signal |
| --- | --- | --- |
| `moire_max`, `moire_mean`, `moire_p90` | Peak-to-mean ratio in the mid-frequency FFT band, computed per-tile on a high-pass residual (gray − Gaussian blur), aggregated across the 12 most-textured tiles | **Screen >> real** — sharp grid peak |
| `hf_ratio` | Fraction of total spectral energy above 25% of Nyquist | Real > screen — reshoots blur fine detail |
| `banding` | Peak/mean of the row-mean FFT | Real > screen — reshoots smear scan-line periodicity |
| `residual_std` | Std of (gray − Gaussian blur, σ=2) | Real > screen — same reason |
| `residual_kurt` | Excess kurtosis of that residual | Real > screen — natural noise is heavier-tailed than reshoot noise |

The moiré features run FFT on the **high-pass residual** rather than raw pixels. This suppresses scene-content periodicity (curtains, water ripples) that would otherwise produce false moiré peaks indistinguishable from a screen grid. Colour features (white balance, saturation, glare fraction) were ablated and dropped — they showed no class separation on this dataset.

## Dataset

100 photos taken with a OnePlus 11: 50 real (mixed lighting, content, indoor/outdoor), 50 reshoots of those reals onto three deliberately different screens — HP Pavilion (~100 PPI, easy moiré), MacBook 14" M2 (254 PPI, medium), OnePlus 11 (525 PPI AMOLED, very hard). The PPI spread was intentional, to *simulate* unseen displays at evaluation.

## Accuracy — honest

| Split | Accuracy | Screen recall |
| --- | --- | --- |
| 5-fold CV (full pool) | 85.0% ± 12.2% | — |
| LOSO, held out HP | 83.8% | 92% |
| LOSO, held out Mac | 81.4% | 89% |
| LOSO, held out OnePlus | 88.9% | **95%** |

**The leave-one-screen-out (LOSO) numbers are the meaningful estimate** for unseen displays: train on two screens, test on the third plus a fixed half-pool of held-out reals. Mean LOSO accuracy 84.7%, mean unseen-screen recall 92%. Published cross-device recapture detection sits at 55–75% (Chen et al., IEEE TDSC 2025; DAST-DG), so 92% recall on a held-out high-PPI screen is genuinely respectable.

The 5-fold CV number (85%) is higher only because it lets the model see *some* photos from every screen during training. Reviewers should treat LOSO as the real number.

## What works, what doesn't

**Works trivially:** loud-moiré screens (HP at low PPI, photographed at close distance).
**Hardest cases (where the 16% errors live):**
1. **Smooth-content reshoots** — a screen displaying a featureless sky or a dark frog photo. Moiré needs detail to develop; without it, screens look like reals.
2. **Naturally periodic reals** — bus interiors with seat-back stripes, woven curtains, water ripples. These produce real frequency peaks indistinguishable from a screen grid.

Both failure modes have the same root cause: every feature here depends on content texture. There's a physical ceiling to texture-based features that only a content-aware approach can break through.

## Latency

| Stage | ms |
| --- | --- |
| Cold-start import (one-time) | ~1100 |
| Per-image inference, warm | mean 104, median 107, p95 115 |

Measured on MacBook M2 Pro CPU, single-threaded, no GPU. FFT on the full-resolution image dominates; the SVM itself is sub-millisecond.

## Cost per image at scale

- **On-device:** free. Pure NumPy + Pillow, no model framework, `model.pkl` is 6 KB.
- **Cloud:** ~10 images/sec on a t4g.small (~$0.017/hr, 2 vCPU). **~$0.0005 per 1000 images, ~$0.50 per million.** Assumes no batching and a single core.

## Trade-off

This lands on the small/fast/cheap end of the design space and trades absolute accuracy for honesty, interpretability, and zero deployment friction. A CNN trained on a larger labelled dataset would push accuracy higher (literature shows 95%+ achievable with Laplacian-CNN patch voting), but at the cost of explainability, training data scale, and on-device size. Given the 100-image constraint and the rubric's stated preferences, classical CV was the right call.

## What I'd improve with more time

1. **Residual-domain CNN on 64×64 patches** with patch voting (Yang & Ni). Literature suggests ~95% single-dataset there. Stays small.
2. **Boundary-consistency test** (Kunina et al. 2023): screen moiré leaks across object boundaries because it's global, natural texture is bounded to objects. Reported 95% precision on DLC-2021. Half-day implementation.
3. **Domain-generalisation training** (SADG, arXiv 2110.03496): adversarial loss on screen identity, scale-space alignment. Designed for cross-device collapse, which is exactly our hard case.
4. **More data on the hard end** — more high-PPI screens (Pixel, iPhone Pro), more flat/smooth content reshoots, more natural-periodicity reals (curtains, ripples) so the classifier can learn the distinction we currently can't make.

## Bonus prompts

**Keeping it accurate as cheaters adapt.** Log scores and metadata (device, time, score distribution) in production. Retrain monthly on disputed/wrong cases. Watch for score distribution drift toward 0.5 — that usually means a new screen class is slipping through. Maintain an adversarial reshoot set that grows with reported fraud cases.

**On-device deployment.** Already small: `model.pkl` is 6 KB, dependencies are NumPy + Pillow only. For Android/iOS, the FFT and Gaussian blur both have native equivalents in OpenCV mobile bindings or Accelerate.framework — features.py could be ported in ~200 lines.

**Choosing the cut-off score.** Don't ship a single 0.5 threshold. Plot precision-recall on a held-out set and offer a two-threshold policy: **auto-block above 0.85** (high precision, low false-block rate for honest users), **auto-pass below 0.3**, **manual review between 0.3 and 0.85**. The exact cutoffs follow from Salescode's business cost ratio: how many false blocks per missed cheater is acceptable.
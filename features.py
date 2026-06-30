"""
Feature extractor for the real-vs-screen-recapture detector.
Imported by both the training notebook and predict.py, so they can't drift.
Deps: numpy + Pillow only.
"""

import numpy as np
from PIL import Image, ImageOps, ImageFilter

TILE      = 640
MAX_TILES = 12
CROP      = 1024
COLOR_MAX = 512
LBP_SIZE  = 512    # LBP works at the scale of the screen pixel, not the camera pixel
RES_SIGMA = 1.0    # high-pass sigma for the moire FFT residual

# Column order is contractual: predict.py rebuilds the vector in this order.
FEATURE_ORDER = [
    "moire_max",
    "moire_mean",
    "moire_p90",
    "hf_ratio",
    "banding",
    "lbp_entropy",     # micro-texture diversity; drops on screen recaptures
    "wb_rg",
    "wb_bg",
    "colorfulness",
    "sat_mean",
    "sat_std",
    "sat_skew",        # shape of the saturation distribution, not its magnitude
    "val_mean",
    "glare_frac",
    "clip_frac",
    "residual_std",
    "residual_kurt",
]


def load_rgb(path):
    # exif_transpose so portrait shots aren't sideways.
    img = ImageOps.exif_transpose(Image.open(path))
    return img.convert("RGB")


# --- fft plumbing --------------------------------------------------------
_hann = {}
def _window(shape):
    # Hann window to suppress edge leakage in the FFT.
    if shape not in _hann:
        h, w = shape
        _hann[shape] = np.outer(np.hanning(h), np.hanning(w))
    return _hann[shape]

_rad = {}
def _radius(shape):
    if shape not in _rad:
        h, w = shape
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
        _rad[shape] = r / r.max()
    return _rad[shape]

def _fft_mag(a):
    a = (a - a.mean()) * _window(a.shape)
    return np.abs(np.fft.fftshift(np.fft.fft2(a)))

def _center_crop(a, size):
    h, w = a.shape
    s = min(size, h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return a[y:y + s, x:x + s]


# --- moire (residual-domain) --------------------------------------------
def _highpass(gray, sigma=RES_SIGMA):
    # gray - gaussian_blur(gray, sigma): suppresses scene content, keeps the screen grid.
    pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
    return gray - blur

def _peak_ratio(tile):
    # Screen grids produce a sharp peak in the mid-frequency band; natural texture spreads flat.
    mag = _fft_mag(tile)
    r = _radius(mag.shape)
    band = mag[(r > 0.15) & (r < 0.45)]
    return float(band.max() / (band.mean() + 1e-8)) if band.size else 0.0

def _detail(tile):
    # Fine-texture energy. Used to pick the most informative tiles for the FFT.
    return np.abs(np.diff(tile, axis=0)).std() + np.abs(np.diff(tile, axis=1)).std()

def _moire(gray):
    # Blur once globally so the kernel is consistent across tiles.
    residual = _highpass(gray, RES_SIGMA)

    h, w = residual.shape
    tiles = [residual[y:y + TILE, x:x + TILE]
             for y in range(0, h - TILE + 1, TILE)
             for x in range(0, w - TILE + 1, TILE)]
    if not tiles:
        s = (min(h, w) // 2) * 2
        tiles = [_center_crop(residual, s)] if s >= 64 else []
    if not tiles:
        return {"moire_max": 0.0, "moire_mean": 0.0, "moire_p90": 0.0}

    tiles.sort(key=_detail, reverse=True)
    peaks = np.array([_peak_ratio(t) for t in tiles[:MAX_TILES]])
    return {
        "moire_max":  float(peaks.max()),
        "moire_mean": float(peaks.mean()),
        "moire_p90":  float(np.percentile(peaks, 90)),
    }


# --- whole-image frequency ----------------------------------------------
def _global_fft(gray):
    c = _center_crop(gray, CROP)
    mag = _fft_mag(c)
    r = _radius(mag.shape)
    hf = float(mag[r > 0.25].sum() / (mag.sum() + 1e-8))

    # Row-mean FFT picks up horizontal banding (scan-line / refresh artifacts).
    rows = c.mean(axis=1)
    rows = rows - rows.mean()
    R = np.abs(np.fft.rfft(rows))
    if R.size > 1:
        R[0] = 0.0
    banding = float(R.max() / (R.mean() + 1e-8))
    return {"hf_ratio": hf, "banding": banding}


# --- LBP entropy --------------------------------------------------------
def _lbp_entropy(gray, size=LBP_SIZE):
    # 8-neighbour LBP at radius 1, then Shannon entropy of the 256-bin code histogram.
    # Real photos carry diverse micro-texture (high entropy); screen recaptures get
    # regularised by the LCD subpixel grid and reshoot blur, so entropy drops.
    pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    pil.thumbnail((size, size))
    g = np.asarray(pil, dtype=np.int32)

    center = g[1:-1, 1:-1]
    code = np.zeros_like(center)
    shifts = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i, (dy, dx) in enumerate(shifts):
        neigh = g[1+dy:g.shape[0]-1+dy, 1+dx:g.shape[1]-1+dx]
        code += ((neigh >= center).astype(np.int32) << i)

    hist, _ = np.histogram(code, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return {"lbp_entropy": float(-np.sum(hist * np.log2(hist)))}


# --- colour --------------------------------------------------------------
def _colour(rgb):
    small = rgb.copy()
    small.thumbnail((COLOR_MAX, COLOR_MAX))
    a = np.asarray(small, dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]

    # Hasler-Susstrunk colourfulness.
    rg = R - G
    yb = 0.5 * (R + G) - B
    colourful = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))

    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    S, V = hsv[..., 1], hsv[..., 2]

    s_mean, s_std = float(S.mean()), float(S.std())
    sat_skew = float(((S - s_mean) ** 3).mean() / (s_std ** 3 + 1e-8))

    return {
        "wb_rg":        float(R.mean() / (G.mean() + 1e-6)),
        "wb_bg":        float(B.mean() / (G.mean() + 1e-6)),
        "colorfulness": colourful,
        "sat_mean":     s_mean,
        "sat_std":      s_std,
        "sat_skew":     sat_skew,
        "val_mean":     float(V.mean()),
        "glare_frac":   float(((R > 245) & (G > 245) & (B > 245)).mean()),
        "clip_frac":    float(((R > 250) | (G > 250) | (B > 250)).mean()),
    }


# --- residual texture ----------------------------------------------------
def _residual(rgb):
    # High-pass texture energy + its kurtosis. Reshooting blurs detail, so reals run higher.
    g = rgb.convert("L")
    res = (np.asarray(g, dtype=np.float32)
           - np.asarray(g.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32))
    std = float(res.std())
    if std < 1e-6:
        kurt = 0.0
    else:
        z = (res - res.mean()) / std
        kurt = float((z ** 4).mean() - 3.0)
    return {"residual_std": std, "residual_kurt": kurt}


def extract_features(path):
    """Image path -> dict of features in FEATURE_ORDER."""
    rgb = load_rgb(path)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    f = {}
    f.update(_moire(gray))
    f.update(_global_fft(gray))
    f.update(_lbp_entropy(gray))
    f.update(_colour(rgb))
    f.update(_residual(rgb))
    return {k: f[k] for k in FEATURE_ORDER}
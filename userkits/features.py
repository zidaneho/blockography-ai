# features.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from joblib import Parallel, delayed

# ---------- atomic feature blocks ----------

def rgb_color_hist(img_rgb, bins_per_channel=8, normalize=False):
    hist = cv2.calcHist([img_rgb], [0,1,2], None,
                        [bins_per_channel]*3, [0,256,0,256,0,256]).flatten()
    if normalize:
        s = hist.sum()
        if s > 0:
            hist = hist / s
    return hist.astype(np.float32)

def lbp_uniform_hist(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

def edge_density(gray, t1=100, t2=200):
    edges = cv2.Canny(gray, t1, t2)
    return np.float32((edges > 0).mean())

def hsv_stats(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return np.float32(hsv[...,2].mean()), np.float32(hsv[...,1].mean())  # brightness(V), saturation(S)

def hue_hist16(img_rgb):
    """16-bin hue histogram, normalized."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[...,0]  # OpenCV hue: 0..179
    hist = np.histogram(H, bins=16, range=(0,180))[0].astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist  # (16,)

def lab_color_moments(img_rgb):
    """LAB mean/std (6 numbers) — robust color summary."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab.reshape(-1,3)
    means = flat.mean(axis=0)
    stds  = flat.std(axis=0)
    return np.hstack([means, stds]).astype(np.float32)  # (6,)

def biome_ratios(img_rgb):
    """Heuristic ratios: vegetation, water, snow, sand."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]
    total = float(H.size)

    r,g,b = img_rgb[...,0].astype(np.int16), img_rgb[...,1].astype(np.int16), img_rgb[...,2].astype(np.int16)
    exg = 2*g - r - b
    veg  = (exg > 20).sum() / max(total,1.0)

    water = (((H > 90) & (H < 140)) & (S > 40)).sum() / max(total,1.0)
    snow  = ((V > 200) & (S < 25)).sum() / max(total,1.0)
    sand  = (((H >= 15) & (H <= 35)) & (S < 120) & (V > 120)).sum() / max(total,1.0)

    return np.asarray([veg, water, snow, sand], dtype=np.float32) 
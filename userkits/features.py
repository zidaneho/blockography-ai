import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
import mahotas

def feature_check(img):
    # standardize color space first
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Color histogram (8x8x8 = 512) ---
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten()

    # --- LBP texture (P=8 -> 10 bins) ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 8+3), range=(0, 8+2))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # --- Simple extras (3) ---
    edges = cv2.Canny(gray, 100, 200)
    edge_density = (edges > 0).mean()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = hsv[...,2].mean()
    saturation = hsv[...,1].mean()

    # total length = 512 + 10 + 3 = 525
    return np.hstack([hist, lbp_hist, [edge_density, brightness, saturation]]).astype(np.float32)

def color_histogram(img, channels=[0, 1, 2], bins=[8, 8, 8], ranges=[0, 256, 0, 256, 0, 256]):
    hist = cv2.calcHist(img, channels, None, bins, ranges)
    return hist.flatten()

def lbp_texture_features(img, P=8, R=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    return lbp_hist

def find_mean(img):
    return np.mean(img, axis=(0, 1))

def find_stddev(img):
    return np.std(img, axis=(0, 1))

def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def shannon_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    return entropy

def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,2])

def green_pixel_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return np.sum(mask > 0) / mask.size

def hog_features(image):
    # Convert to grayscale for HOG
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(image_gray, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features.tolist()
def haralick_texture_features(image):
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = mahotas.features.haralick(image_gray)
    return features.mean(axis=0).tolist() # Average over 4 directions
def gabor_features(image):
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feats = []
    for theta in (0, 1, 2, 3): # 4 orientations
        for freq in (0.1, 0.5, 0.9): # 3 frequencies
            real, imag = gabor(image_gray, frequency=freq, theta=np.pi/4 * theta)
            feats.append(real.mean())
            feats.append(real.var())
    return feats
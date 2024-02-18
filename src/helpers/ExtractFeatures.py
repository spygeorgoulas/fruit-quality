from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.color import rgb2hsv
from skimage.measure import find_contours


# Function to extract color histograms from the HSV color space
def extract_color_histogram_hsv(image, bins=256):
    hsv_image = rgb2hsv(image)
    hist_hue = cv2.calcHist([img_as_ubyte(hsv_image)], [0], None, [bins], [0, 256]).ravel()
    hist_sat = cv2.calcHist([img_as_ubyte(hsv_image)], [1], None, [bins], [0, 256]).ravel()
    # Normalize histograms
    hist_hue /= hist_hue.sum()
    hist_sat /= hist_sat.sum()
    return np.concatenate([hist_hue, hist_sat])

# Function to extract texture features using Local Binary Patterns
def extract_lbp_features(image, P=8, R=1):
    gray_image = rgb2gray(image)
    gray_image = img_as_ubyte(gray_image)
    lbp = local_binary_pattern(gray_image, P, R, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P * R + 3), range=(0, P * R + 2))
    lbp_hist = lbp_hist.astype('float') / (lbp_hist.sum() + 1e-6)
    return lbp_hist

# Function to extract shape features using contours
def extract_shape_features(image):
    # Convert to grayscale and find contours
    gray_image = rgb2gray(image)
    contours = find_contours(gray_image, level=0.8)
    num_contours = len(contours)
    
    shape_features = [num_contours] 

    return np.array(shape_features)

def extract_all_features(image):
    color_hist_hsv = extract_color_histogram_hsv(image)
    lbp_features = extract_lbp_features(image)
    shape_features = extract_shape_features(image)

    features = np.concatenate([color_hist_hsv, lbp_features, shape_features])
    return features

from joblib import load
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from helpers.ExtractFeatures import extract_all_features
from helpers.RemoveBackground import remove_background
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import cv2
from skimage.color import rgb2hsv
from skimage.measure import find_contours
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path


# Function to load saved model, scaler, and label encoder
def load_model_scaler_encoder(model_path, scaler_path, encoder_path):
    model = load(model_path)
    scaler = load(scaler_path)
    label_encoder = load(encoder_path)
    return model, scaler, label_encoder


# Function to preprocess and predict new data
def predict_new_data(input_features, model, scaler, label_encoder):
    # Determine the shape of the input features
    input_features_shape = input_features.shape

    # Reshape input features appropriately
    if len(input_features_shape) == 1:  # Input features have a single feature
        # Reshape input features to a row vector
        input_features_2d = input_features.reshape(1, -1)
    elif len(input_features_shape) == 2 and input_features_shape[1] == 1:  # Input features have a single sample
        # Reshape input features to a column vector
        input_features_2d = input_features.reshape(-1, 1)
    else:
        # No reshaping needed
        input_features_2d = input_features

    # Normalize features
    input_features_scaled = scaler.transform(input_features_2d)

    # Predict using the model
    predictions = model.predict(input_features_scaled)

    # Decode predictions back to original labels
    decoded_predictions = label_encoder.inverse_transform(predictions)

    return decoded_predictions


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

    # Example feature: number of contours
    num_contours = len(contours)

    shape_features = [num_contours]

    return np.array(shape_features)


def extract_all_features(image):
    color_hist_hsv = extract_color_histogram_hsv(image)
    lbp_features = extract_lbp_features(image)
    shape_features = extract_shape_features(image)

    # Combine features into a single feature vector
    features = np.concatenate([color_hist_hsv, lbp_features, shape_features])
    return features


if __name__ == "__main__":
    # Load model, scaler, and label encoder
    model, scaler, label_encoder = load_model_scaler_encoder('model/svm_model.pkl', 'model/scaler.pkl',
                                                             'model/label_encoder.pkl')

    # Set the path to the folder containing images
    images_folder_path = 'demo_dataset-small'

    true_labels = []
    all_predictions = []

    # Iterate through subfolders in the images folder
    for quality_category in os.listdir(images_folder_path):
        quality_category_path = os.path.join(images_folder_path, quality_category)
        if os.path.isdir(quality_category_path):
            for fruit_category in os.listdir(quality_category_path):
                fruit_category_path = os.path.join(quality_category_path, fruit_category)
                if os.path.isdir(fruit_category_path):
                    # Iterate through images in the fruit category
                    for image_name in os.listdir(fruit_category_path):
                        image_path = os.path.join(fruit_category_path, image_name)
                        if image_name.endswith(".jpg") or image_name.endswith(".png"):
                            # Load image
                            image = io.imread(image_path)
                            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                                continue
                            image = remove_background(image)
                            image = resize(image, (256, 192), anti_aliasing=True)
                            image = img_as_ubyte(image)

                            # Extract features
                            features = extract_all_features(image)

                            # Extract label
                            true_label = fruit_category
                            true_labels.append(true_label)

                            # Predict using the loaded input features
                            predictions = predict_new_data(features, model, scaler, label_encoder)
                            all_predictions.append(predictions)
                            print(f"Predictions for {image_name}: {predictions}")
                        else:
                            continue

    # Evaluate predictions
    true_labels = np.array(true_labels)
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, average='weighted')
    recall = recall_score(true_labels, all_predictions, average='weighted')
    f1 = f1_score(true_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

from joblib import load
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from helpers.ExtractFeatures import extract_all_features
from helpers.RemoveBackground import remove_background
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

# Function to load saved model, scaler, and label encoder
def load_model_scaler_encoder(model_path, scaler_path, encoder_path):
    model = load(model_path)
    scaler = load(scaler_path)
    label_encoder = load(encoder_path)
    return model, scaler, label_encoder

# Function to preprocess and predict new data
def predict_new_data(input_features, model, scaler, label_encoder):
    # Normalize features
    input_features_scaled = scaler.transform([input_features])

    # Predict using the model
    predictions = model.predict(input_features_scaled)
    # Decode predictions back to original labels
    decoded_predictions = label_encoder.inverse_transform(predictions)

    return decoded_predictions

def process_images(input_folder, model, scaler, label_encoder):
    true_labels = []  # This would need to be provided or inferred somehow since it's not part of the input
    all_predictions = []
    # Iterate through images in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
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

            # Predict using the loaded input features
            predictions = predict_new_data(features, model, scaler, label_encoder)
            all_predictions.append(predictions[0])  # Assuming we want the first prediction for simplicity
            print(f"Predictions for {image_name}: {predictions}")
        else:
            continue
    return

def process_features(input_folder, model, scaler, label_encoder):
    true_labels = []
    all_predictions = []

    # Iterate through .npy files in the features folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            features_file_path = os.path.join(input_folder, filename)
            # Load features from .npy file
            features_dict = np.load(features_file_path, allow_pickle=True).item()  # Load as dictionary
            input_features = features_dict['features']  # Extract features from dictionary
            true_label = features_dict['label']  # Extract true label from dictionary
            true_labels.append(true_label)
            # Predict using the loaded input features
            predictions = predict_new_data(input_features, model, scaler, label_encoder)
            all_predictions.append(predictions)
            print(f"Predictions for {filename}: {predictions}")
        else:
            continue
    return true_labels, all_predictions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels from images or pre-extracted features.")
    parser.add_argument('--mode', type=str, choices=['images', 'features'], required=True,
                        help='Operational mode: "images" to process images, "features" to process pre-extracted features.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the input folder containing either images or pre-extracted features.')

    args = parser.parse_args()

    # Load model, scaler, and label encoder
    model, scaler, label_encoder = load_model_scaler_encoder('src/model/svm_model.pkl', 'src/model/scaler.pkl', 'src/model/label_encoder.pkl')
    
    if args.mode == "images":
        process_images(args.input_folder, model, scaler, label_encoder)
        exit()
    elif args.mode == "features":
        true_labels, all_predictions = process_features(args.input_folder, model, scaler, label_encoder)
    
    # Evaluate predictions
    accuracy = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, average='weighted')
    recall = recall_score(true_labels, all_predictions, average='weighted')
    f1 = f1_score(true_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
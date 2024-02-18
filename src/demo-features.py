from joblib import load
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from helpers.ExtractFeatures import extract_all_features
from helpers.RemoveBackground import remove_background
import argparse
from joblib import load
import os
import numpy as np
from joblib import load
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

if __name__ == "__main__":
    # Load model, scaler, and label encoder
    model, scaler, label_encoder = load_model_scaler_encoder('model/svm_model.pkl', 'model/scaler.pkl', 'model/label_encoder.pkl')

    # Set the path to the folder containing input features
    features_folder_path = 'features_demo'
    true_labels = []
    all_predictions = []

    # Iterate through .npy files in the features folder
    for filename in os.listdir(features_folder_path):
        if filename.endswith(".npy"):
            features_file_path = os.path.join(features_folder_path, filename)
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

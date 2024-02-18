from joblib import load
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from helpers.ExtractFeatures import extract_all_features
from helpers.RemoveBackground import remove_background
import argparse

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

# Load model, scaler, and label encoder
model, scaler, label_encoder = load_model_scaler_encoder('model/svm_model.pkl', 'model/scaler.pkl', 'model/label_encoder.pkl')
resize_shape=(256, 192)

if __name__ == "__main__":
    # Define the argument parser to read in the input and output folder paths
    parser = argparse.ArgumentParser(description='Preprocess and augment fruit images')
    parser.add_argument('-i', '--input_folder_path', type=str, required=True, help='Path to the dataset folder')

    # Parse the arguments
    args = parser.parse_args()
    
    ### MODIFY BELLOW ####
    #read from folder
    
from joblib import load
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np
import sys

# Ensure to define or import your extract_all_features function here

# Function to load saved model, scaler, and label encoder
def load_model_scaler_encoder(model_path, scaler_path, encoder_path):
    model = load(model_path)
    scaler = load(scaler_path)
    label_encoder = load(encoder_path)
    return model, scaler, label_encoder

# Function to preprocess and predict new data
def predict_new_data(input_features, model, scaler, label_encoder):
    input_features_scaled = scaler.transform([input_features])
    predictions = model.predict(input_features_scaled)
    decoded_predictions = label_encoder.inverse_transform(predictions)

    return decoded_predictions

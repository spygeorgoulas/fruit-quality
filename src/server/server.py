from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import json
from json.decoder import JSONDecodeError
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from joblib import load
from ..helpers.ExtractFeatures import extract_all_features
from ..helpers.RemoveBackground import remove_background


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
    

app = Flask(__name__, template_folder='templates', static_folder='static')


UPLOAD_FOLDER = 'src/server/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

   
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image_file = request.files['image']
        # Predict the image to get fruit name and quality
        label = predict(image_file)  # Adjust the predict function accordingly
        label_string = label.split('_')
        fruit_name = label_string[0]
        quality = label_string[1]
        # Set the path for the new filename using fruit name
        new_filename = f"{fruit_name}.jpg"
        new_image_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        # Save the image with the new name
        # Ensure we are saving the file in binary format
        image_file.stream.seek(0)  # Reset the stream pointer to the beginning of the file
        with open(new_image_path, 'wb') as f:
            f.write(image_file.stream.read()) 
            
        # Load existing qualities or create a new one
        qualities_path = os.path.join(app.config['UPLOAD_FOLDER'], 'qualities.json')
        try:
            if os.path.exists(qualities_path):
                with open(qualities_path, 'r') as file:
                    qualities = json.load(file)
            else:
                qualities = {}
        except JSONDecodeError:
            qualities = {}

        # Update the qualities file
        qualities[fruit_name] = quality
        with open(qualities_path, 'w') as file:
            json.dump(qualities, file)

        return jsonify({'message': 'Image uploaded successfully', 'image_path': new_image_path, 'label': quality})

    return jsonify({'message': 'No image provided'})


@app.route('/home')
def display_all_images():
    image_files = os.listdir(app.config['UPLOAD_FOLDER'])
    image_names = [os.path.basename(filename) for filename in image_files if filename.endswith('.jpg')]  # Ensure only images are listed

    # Load qualities.json
    qualities_path = os.path.join(app.config['UPLOAD_FOLDER'], 'qualities.json')
    with open(qualities_path, 'r') as file:
        image_qualities = json.load(file)

    # Filter out images without a quality entry (optional)
    image_names = [name for name in image_names if name.split('.')[0] in image_qualities]
    print(image_names)
    print(image_qualities)
    return render_template('page.html', image_names=image_names, image_qualities=image_qualities)
    
# Define a route to handle a POST request for the site
@app.route('/post_site', methods=['POST'])
def handle_post_request_for_site():
    data = request.form.get('data')  # Assuming 'data' is the field name in your form
    # Process the data as needed
    return jsonify({'message': 'Data received successfully from the site', 'data': data})

def predict(image_path):
    resize_shape=(256, 192)
    image = io.imread(image_path)
    image = remove_background(image)
    image = resize(image, resize_shape, anti_aliasing=True)
    image = img_as_ubyte(image)
    features = extract_all_features(image)
    model, scaler, label_encoder = load_model_scaler_encoder('src/model/svm_model.pkl', 'src/model/scaler.pkl', 'src/model/label_encoder.pkl')
    predicted_label = predict_new_data(features, model, scaler, label_encoder)
    return predicted_label[0]

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=6060)


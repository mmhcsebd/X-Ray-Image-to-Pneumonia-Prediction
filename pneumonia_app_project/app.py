import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# --- Configuration and Constants ---
app = Flask(__name__)
# Enable CORS to allow the frontend (index.html) to communicate with this server.
CORS(app) 

# --- Model Parameters (MUST MATCH TRAINING) ---
# IMPORTANT: Ensure IMG_SIZE matches the size your model was trained on (e.g., 150x150)
IMG_SIZE = 150
# LABELS[0] is for probability < 0.5 (Normal), LABELS[1] is for probability > 0.5 (Pneumonia)
LABELS = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = 'custom_cnn_pneumonia_model.h5'

# --- Model Loading (Global) ---
# Load the model only once when the server starts
try:
    if not os.path.exists(MODEL_PATH):
        # If the model file is not found, print an error and set model to None
        print(f"❌ Error: Model file not found at: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    # --- DIAGNOSTIC ADDITION ---
    # Print the absolute path to confirm location
    print(f"📁 Looking for model at: {os.path.abspath(MODEL_PATH)}")
    # ---------------------------
    
    # Suppress TensorFlow warning output for cleaner console
    tf.get_logger().setLevel('ERROR') 
    
    # Load the Keras model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully and is ready for predictions.")

except Exception as e:
    print(f"❌ Critical Error during model loading: {e}")
    model = None

# --- Helper Function for Preprocessing ---
def preprocess_image(image_bytes):
    """
    Takes image file bytes, converts it to a grayscale NumPy array, 
    resizes it to IMG_SIZE, and normalizes pixel values.
    """
    # 1. Open image from bytes using PIL
    image_stream = io.BytesIO(image_bytes)
    # Convert to Grayscale ('L')
    img = Image.open(image_stream).convert('L') 

    # 2. Convert PIL image to NumPy array
    img_array = np.array(img)

    # 3. Resize and Normalize using OpenCV
    resized_image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    normalized_image = resized_image / 255.0

    # 4. Reshape for model input (Batch, Height, Width, Channels)
    # Channels=1 for grayscale
    input_image = normalized_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return input_image

# --- API Route for Status Check (New addition) ---
@app.route('/', methods=['GET'])
def home():
    """Returns a simple message to confirm the API server is running."""
    return jsonify({
        "status": "online",
        "message": "Pneumonia Prediction API is running. Use /predict POST endpoint for predictions."
    })


# --- API Route for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint that receives the image file, performs prediction, and returns JSON.
    """
    if model is None:
        return jsonify({"error": "The prediction service is unavailable because the model failed to load."}), 503

    # 1. Check for file in request
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided in the request payload."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Read image data
            image_bytes = file.read()
            
            # Preprocess the image
            input_tensor = preprocess_image(image_bytes)

            # Make prediction (returns probability of the positive class - Pneumonia)
            # [0][0] extracts the single float probability
            prediction_prob = model.predict(input_tensor, verbose=0)[0][0] 

            # Determine class label (Binary classification: PNEUMONIA > 0.5)
            # index 1 is PNEUMONIA, index 0 is NORMAL
            predicted_index = 1 if prediction_prob > 0.5 else 0
            predicted_label = LABELS[predicted_index]
            
            # Confidence is the probability of the predicted class
            confidence = prediction_prob if predicted_label == 'PNEUMONIA' else 1.0 - prediction_prob

            # Return results as JSON
            return jsonify({
                "prediction": predicted_label,
                # Convert numpy float to standard Python float for JSON serialization
                "probability": float(confidence) 
            })

        except Exception as e:
            print(f"Prediction processing error: {e}")
            return jsonify({"error": f"Internal server error during image processing or prediction: {e}"}), 500

# --- Server Run ---
if __name__ == '__main__':
    # Running on 127.0.0.1:5000 matches the API_ENDPOINT in your index.html
    print("Starting Flask server...")
    app.run(host='127.0.0.1', port=5000)

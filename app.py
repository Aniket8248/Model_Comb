from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# Load both models
exception_model = load_model('Exception_model_20Nov.h5')
blood_cancer_model = load_model('xception_blood_cancer_model.h5')

# Route to serve the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')  # Serve the index.html from the current directory

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            try:
                # Save uploaded file temporarily
                temp_filename = f"{uuid.uuid4()}.jpg"
                file.save(temp_filename)
                
                # Preprocess the image
                img = image.load_img(temp_filename, target_size=(256, 256))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img / 255.0
                
                # Step 1: Pass the image to the exception model
                exception_prediction = exception_model.predict(img)
                exception_result = "Accept" if exception_prediction[0][0] <= 0.5 else "Not Accept"
                
                # If "Not Accept," return the result and stop further processing
                if exception_result == "Not Accept":
                    os.remove(temp_filename)  # Clean up
                    return jsonify({
                        'stage': 'Exception Model',
                        'prediction': exception_result,
                        'confidence': float(exception_prediction[0][0])
                    })
                
                # Step 2: Pass the image to the blood cancer model
                cancer_prediction = blood_cancer_model.predict(img)
                cancer_result = "Cancer Detected" if cancer_prediction[0][0] > 0.5 else "No Cancer Detected"
                cancer_confidence = cancer_prediction[0][0] if cancer_prediction[0][0] > 0.5 else 1 - cancer_prediction[0][0]
                
                # Clean up: Remove the temporary image file
                os.remove(temp_filename)
                
                # Return the result as JSON
                return jsonify({
                    'stage': 'Blood Cancer Model',
                    'prediction': cancer_result,
                    'confidence': float(cancer_confidence)
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model file path
MODEL_PATH = 'best_model_vgg19.keras'
model = None  # Delay loading the model

# Prediction function
def load_model_once():
    global model
    if model is None:
        model = load_model(MODEL_PATH)

def predict_image(img_path):
    load_model_once()  # Ensure model is loaded only when needed
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    if 0.4 < prediction < 0.6:
        return "Uncertain - Please upload a proper mouth image"
    elif prediction > 0.5:
        return "Prediction: NON-CANCER"
    else:
        return "Prediction: CANCER"

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    img_path = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction_text = predict_image(file_path)
            img_path = file_path

    return render_template('index.html', prediction=prediction_text, img_path=img_path)

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))  # Render will set PORT env var
    app.run(host='0.0.0.0', port=port)

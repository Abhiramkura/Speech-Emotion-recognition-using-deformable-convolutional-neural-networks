from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
import pickle
from feature_extractor import extract_features

UPLOAD_FOLDER = 'uploads'  
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  

model = tf.keras.models.load_model('model/dcnn_model.h5')

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'audio' not in request.files or request.files['audio'].filename == '':
            flash("Please upload a .wav file before predicting.", 'error')  
            return redirect(request.url)

        file = request.files['audio']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            features = extract_features(filepath)
            features = np.expand_dims(features, axis=0)

            prediction = model.predict(features)
            predicted_index = np.argmax(prediction)
            emotion = label_encoder.classes_[predicted_index]

            return render_template('index.html', prediction=emotion, filename=filename, audio_name=filename)

        else:
            flash("Invalid file format. Please upload a .wav file.", 'error')  
            return redirect(request.url)

    return render_template('index.html', prediction=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
    app.run(debug=True)

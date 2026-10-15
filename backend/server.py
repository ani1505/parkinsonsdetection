from flask import Flask, request, jsonify, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
import os
import numpy as np
from werkzeug.utils import secure_filename
import bcrypt

from preprocessing import preprocess_video
from feature_extraction import extract_optical_flow
from svm_model import train_svm_model, predict_svm_model

app = Flask(__name__, instance_relative_config=True, static_folder='frontend/static', template_folder='frontend/templates')

# Configuration from config.py
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATASET_FOLDER = 'dataset'
PARKINSONS_FOLDER = 'parkinsons_videos'
NON_PARKINSONS_FOLDER = 'non_parkinsons_videos'

svm_model = None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    else:
        return jsonify({'error': 'Could not upload file'})

@app.route('/predict', methods=['POST'])
def predict():
    video_file = request.json.get('filename')
    if not video_file:
        return jsonify({'error': 'No filename provided'})
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file)
    
    # Preprocess the video and extract features
    frames = preprocess_video(video_path)
    max_length = 1000  # Set the maximum length for feature vectors
    features = extract_optical_flow(frames, max_length)
    
    # Convert features list to NumPy array and reshape
    features_array = np.array(features)
    features_reshaped = features_array.reshape(1, -1)
    
    # Use the trained SVM model to make a prediction
    prediction = predict_svm_model(svm_model, features_reshaped)
    
    # Map prediction value to label
    prediction_label = "Positive" if prediction == 1 else "Negative"
    
    return jsonify({'prediction': prediction_label})

@app.route('/')
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/register', methods=['GET', 'POST'])
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if not name or not email or not password:
            return render_template('welcome.html', error='All fields are required')

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/welcome')

    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not email or not password:
            return render_template('welcome.html', error='All fields are required')

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/index')
        else:
            return render_template('welcome.html', error='Invalid user')

    return render_template('welcome.html')

@app.route('/index')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('index.html', user=user)

    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/welcome')

if __name__ == '__main__':
    # Load dataset and train SVM model
    X_train, y_train = [], []  # Initialize empty lists for features and labels

    # Load Parkinson's videos
    for filename in os.listdir(os.path.join(DATASET_FOLDER, PARKINSONS_FOLDER)):
        video_path = os.path.join(DATASET_FOLDER, PARKINSONS_FOLDER, filename)
        frames = preprocess_video(video_path)
        max_length = 1000  # Set the maximum length for feature vectors
        features = extract_optical_flow(frames, max_length)
        X_train.append(features)
        y_train.append(1)  # 1 for Parkinson's videos

    # Load non-Parkinson's videos
    for filename in os.listdir(os.path.join(DATASET_FOLDER, NON_PARKINSONS_FOLDER)):
        video_path = os.path.join(DATASET_FOLDER, NON_PARKINSONS_FOLDER, filename)
        frames = preprocess_video(video_path)
        max_length = 1000  # Set the maximum length for feature vectors
        features = extract_optical_flow(frames, max_length)
        X_train.append(features)
        y_train.append(0)  # 0 for non-Parkinson's videos

    # Convert feature vectors to 2D array
    X_train = np.array(X_train)

    # Reshape X_train to 2D array if needed
    if len(X_train.shape) == 3:
        n_samples = X_train.shape[0]
        X_train = X_train.reshape(n_samples, -1)

    # Train SVM model
    svm_model = train_svm_model(X_train, y_train)
    
    app.run(debug=True)

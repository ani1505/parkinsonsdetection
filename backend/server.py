from flask import Flask, request, jsonify, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
import os
import numpy as np
from werkzeug.utils import secure_filename
import bcrypt
from pathlib import Path

from preprocessing import preprocess_video
from feature_extraction import extract_optical_flow
from svm_model import train_svm_model, predict_svm_model

app = Flask(__name__, instance_relative_config=True, static_folder='frontend/static', template_folder='frontend/templates')

# Configuration from config.py
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'local-only-change-this-key')
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:////tmp/parkinsonsdetection.db' if os.getenv('RENDER') else 'sqlite:///database.db')

db = SQLAlchemy(app)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/parkinsons-uploads' if os.getenv('RENDER') else 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATASET_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
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
        if Path(file.filename).suffix.lower() not in {'.mp4', '.mov', '.webm', '.avi', '.mkv'}:
            return jsonify({'error': 'Unsupported video file type'}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    else:
        return jsonify({'error': 'Could not upload file'})

@app.route('/predict', methods=['POST'])
def predict():
    video_file = (request.get_json(silent=True) or {}).get('filename')
    if not video_file:
        return jsonify({'error': 'No filename provided'}), 400
    if video_file != os.path.basename(video_file):
        return jsonify({'error': 'Invalid filename'}), 400
    if svm_model is None:
        return jsonify({'error': 'Model not loaded. Start the server using python backend/server.py'}), 503
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file)
    
    if not os.path.isfile(video_path):
        return jsonify({'error': 'Video file does not exist'}), 404
    # Preprocess the video and extract features
    frames = preprocess_video(video_path)
    if len(frames) < 2:
        return jsonify({'error': 'Video has too few sampled frames'}), 422
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
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if not name or not email or not password:
            return render_template('welcome.html', error='All fields are required')

        new_user = User(name=name, email=email, password=password)
        try:
            db.session.add(new_user)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return render_template('welcome.html', error='Email already registered'), 409
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
        if user:
            return render_template('index.html', user=user)

    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/welcome')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
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
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '10000')), debug=False)

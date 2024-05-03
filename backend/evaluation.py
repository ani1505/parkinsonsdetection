import os
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from preprocessing import preprocess_video
from feature_extraction import extract_optical_flow
from svm_model import train_svm_model
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_FOLDER = 'dataset'
PARKINSONS_FOLDER = 'parkinsons_videos'
NON_PARKINSONS_FOLDER = 'non_parkinsons_videos'

# Load dataset and split into training and test sets
def load_data_and_split():
    X, y = [], []  # Initialize empty lists for features and labels

    # Load Parkinson's videos
    for filename in os.listdir(os.path.join(DATASET_FOLDER, PARKINSONS_FOLDER)):
        video_path = os.path.join(DATASET_FOLDER, PARKINSONS_FOLDER, filename)
        frames = preprocess_video(video_path)
        max_length = 1000  # Set the maximum length for feature vectors
        for features in extract_optical_flow(frames, max_length):
            X.append(features)
            y.append(1)  # 1 for Parkinson's videos

    # Load non-Parkinson's videos
    for filename in os.listdir(os.path.join(DATASET_FOLDER, NON_PARKINSONS_FOLDER)):
        video_path = os.path.join(DATASET_FOLDER, NON_PARKINSONS_FOLDER, filename)
        frames = preprocess_video(video_path)
        max_length = 1000  # Set the maximum length for feature vectors
        for features in extract_optical_flow(frames, max_length):
            X.append(features)
            y.append(0)  # 0 for non-Parkinson's videos

    # Convert feature vectors and labels to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Load dataset and split into training and test sets
    X_train, X_test, y_train, y_test = load_data_and_split()

    # Train SVM model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svm_model = train_svm_model(X_train_scaled, y_train)

    # Evaluate the model
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test set: {accuracy:.2f}')

    # Plot Confusion Matrix
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Plot ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    plt.show()

    # Plot Feature Importance
    if hasattr(svm_model, 'coef_'):
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(svm_model.coef_[0])), svm_model.coef_[0])
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient')
        plt.title('Feature Importance')
        plt.show()

    # Plot Learning Curve
    train_sizes, train_scores, valid_scores = learning_curve(svm_model, X_train_scaled, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
    plt.plot(train_sizes, np.mean(valid_scores, axis=1), 'o-', label='Validation Score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

    # Plot Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(y_train)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

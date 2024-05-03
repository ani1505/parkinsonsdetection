# svm_model.py
from sklearn.svm import SVC

def train_svm_model(X_train, y_train):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    return svm_model

def predict_svm_model(svm_model, features):
    prediction = svm_model.predict(features.reshape(1, -1))
    return prediction[0]

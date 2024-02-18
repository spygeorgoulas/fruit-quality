from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from joblib import dump
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def load_data(directory):
    features = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            file_path = Path(directory) / file
            data = np.load(file_path, allow_pickle=True).item()
            features.append(data['features'])
            labels.append(data['label'])
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Define the argument parser to read in the input and output folder paths
    parser = argparse.ArgumentParser(description='Preprocess and extract features of given folder')
    parser.add_argument('-train', '--train_features_folder_path', type=str, required=True, help='Path to the train features folder')
    parser.add_argument('-test', '--test_features_folder_path', type=str, required=True, help='Path to the test features folder')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Load training and testing data
    train_data_dir = args.train_features_folder_path
    test_data_dir = args.test_features_folder_path
    
    X_train, y_train = load_data(train_data_dir)
    X_test, y_test = load_data(test_data_dir)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    clf = svm.SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    clf.fit(X_train, y_train) 


    # Predictions and metrics
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.2f}\nPrecision (Weighted): {precision:.2f}\nRecall (Weighted): {recall:.2f}\nF1 Score (Weighted): {f1:.2f}")

    # Classification report
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Save the best model, scaler, and label encoder
    dump(clf, 'src/model/svm_model.pkl')
    dump(scaler, 'src/model/scaler.pkl')
    dump(label_encoder, 'src/model/label_encoder.pkl')
    print("Best model, scaler, and label encoder saved.")
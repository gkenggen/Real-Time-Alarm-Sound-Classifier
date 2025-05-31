
import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

# --- USER CONFIGURATION --- #
# Path to your alarm dataset (each subfolder is a class name)
DATA_DIR = "alarm_dataset"

# Output path for the trained model
MODEL_OUTPUT_PATH = "alarm_model.pkl"

# List of feature extraction functions
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None, duration=5.0)
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        # Compute Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        chroma_mean = np.mean(chroma, axis=1)
        # Compute Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        # Compute Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        # Combine all features into one vector
        features = np.hstack([mfccs_mean, chroma_mean, contrast_mean, zcr_mean])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -------------------------------- #
if __name__ == "__main__":
    X = []
    y = []
    # Iterate over each class folder
    for label in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, label)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".wav"):
                    file_path = os.path.join(folder_path, filename)
                    feats = extract_features(file_path)
                    if feats is not None:
                        X.append(feats)
                        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"Extracted features for {len(X)} files across {len(set(y))} classes.")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define a Random Forest classifier with GridSearch
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)

    print("Training Random Forest with GridSearchCV...")
    clf.fit(X_train, y_train)

    print("Best parameters:", clf.best_params_)
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the best model
    joblib.dump(clf.best_estimator_, MODEL_OUTPUT_PATH)
    print(f"Trained model saved to {MODEL_OUTPUT_PATH}")

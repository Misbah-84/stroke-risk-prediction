import os
import pickle
import numpy as np
from sklearn.metrics import classification_report

class PreFittedSoftVotingEnsemble:
    def __init__(self, estimators):
        self.estimators = estimators

    def predict_proba(self, X):
        probs = [est.predict_proba(X) for est in self.estimators]
        return np.mean(probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def run_ensemble():
    cwd = os.getcwd()
    
    # 1. Load data
    processed_data_path = os.path.join(cwd, "data", "processed_data.pkl")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}. Run data_prep.py first.")

    with open(processed_data_path, "rb") as f:
        data = pickle.load(f)

    X_test_scaled = data["X_test_scaled"]
    y_test = data["y_test"]
    scaler = data["scaler"]

    # 2. Load pre-trained models
    models_dir = os.path.join(cwd, "models")
    dt_path = os.path.join(models_dir, "dt_model.pkl")
    knn_path = os.path.join(models_dir, "knn_model.pkl")
    svc_path = os.path.join(models_dir, "svc_model.pkl")

    if not (os.path.exists(dt_path) and os.path.exists(knn_path) and os.path.exists(svc_path)):
        raise FileNotFoundError("One or more trained model files are missing from 'models/'. Run the training scripts first.")

    with open(dt_path, "rb") as f:
        dt = pickle.load(f)
    with open(knn_path, "rb") as f:
        knn = pickle.load(f)
    with open(svc_path, "rb") as f:
        svc = pickle.load(f)

    print("Loaded all three pre-trained models successfully.")

    # 3. Construct and evaluate ensemble
    ensemble = PreFittedSoftVotingEnsemble([dt, knn, svc])
    
    y_pred = ensemble.predict(X_test_scaled)
    print("\nEnsemble (Soft-Voting) Classification Report on Test Partition:")
    print(classification_report(y_test, y_pred))

    # 4. Export the serialized ensemble and scaler
    artifact = {
        'model': ensemble,
        'scaler': scaler
    }
    
    export_path = os.path.join(models_dir, 'stroke_ensemble_model.pkl')
    with open(export_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"Successfully packaged and serialized ensemble model and scaler into '{export_path}'")

if __name__ == "__main__":
    run_ensemble()

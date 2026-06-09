import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def train():
    cwd = os.getcwd()
    processed_data_path = os.path.join(cwd, "data", "processed_data.pkl")
    
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}. Run data_prep.py first.")

    with open(processed_data_path, "rb") as f:
        data = pickle.load(f)

    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    try:
        from imblearn.over_sampling import SMOTE
        use_smote = True
    except ImportError:
        use_smote = False

    print("Training K-Nearest Neighbors Classifier...")
    if use_smote:
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        print("Fallback: Using weights='distance'")
        model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\nKNN Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save serialized model
    models_dir = os.path.join(cwd, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "knn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved KNN model to: {model_path}\n")

if __name__ == "__main__":
    train()

import os
import pickle
from sklearn.svm import SVC
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

    print("Training Support Vector Classifier...")
    if use_smote:
        model = SVC(probability=True, kernel="rbf", random_state=42)
    else:
        print("Fallback: Using class_weight='balanced'")
        model = SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\nSVC Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save serialized model
    models_dir = os.path.join(cwd, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "svc_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved SVC model to: {model_path}\n")

if __name__ == "__main__":
    train()

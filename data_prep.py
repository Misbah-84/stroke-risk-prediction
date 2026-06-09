import os
import shutil
import zipfile
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data():
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    os.makedirs(data_dir, exist_ok=True)

    target_filename = "healthcare-dataset-stroke-data.csv"
    dest_file_path = os.path.join(data_dir, target_filename)

    # 1. Automate migration if not already in data/
    if not os.path.exists(dest_file_path):
        home_dir = os.path.expanduser("~")
        downloads_dir = os.path.join(home_dir, "Downloads")
        source_file_path = os.path.join(downloads_dir, target_filename)
        
        file_copied = False
        try:
            if os.path.exists(source_file_path):
                print("Found unzipped dataset in Downloads. Copying...")
                shutil.copy(source_file_path, dest_file_path)
                file_copied = True
            else:
                raise FileNotFoundError()
        except Exception:
            archive_path = os.path.join(downloads_dir, "archive.zip")
            if os.path.exists(archive_path) and zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    if target_filename in zip_ref.namelist():
                        print(f"Extracting '{target_filename}' from Downloads/archive.zip...")
                        zip_ref.extract(target_filename, data_dir)
                        file_copied = True
            
        if not file_copied:
            raise FileNotFoundError(f"Could not locate '{target_filename}' in Downloads or data/ folder.")

    # 2. Data Cleaning & Splitting
    print(f"Loading dataset from: {dest_file_path}")
    df = pd.read_csv(dest_file_path)

    # Impute missing bmi with median
    bmi_median = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(bmi_median)
    print(f"Filled missing 'bmi' values with median: {bmi_median}")

    # Set up features and target
    continuous_features = ['age', 'avg_glucose_level', 'bmi']
    binary_features = ['hypertension', 'heart_disease']
    
    X_cont = df[continuous_features]
    X_bin = df[binary_features]
    y = df['stroke']

    # Stratified split (80/20)
    X_train_cont, X_test_cont, X_train_bin, X_test_bin, y_train, y_test = train_test_split(
        X_cont, X_bin, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Fit scaler ONLY on continuous features
    scaler = StandardScaler()
    X_train_cont_scaled = scaler.fit_transform(X_train_cont)
    X_test_cont_scaled = scaler.transform(X_test_cont)
    print("Fitted StandardScaler ONLY on continuous features.")

    # Convert binary features to raw integers (0 and 1)
    X_train_bin_arr = X_train_bin.values
    X_test_bin_arr = X_test_bin.values

    # Horizontally stack scaled continuous and raw binary features
    X_train_scaled = np.hstack((X_train_cont_scaled, X_train_bin_arr))
    X_test_scaled = np.hstack((X_test_cont_scaled, X_test_bin_arr))

    # Apply SMOTE to training partition if imblearn is installed
    try:
        from imblearn.over_sampling import SMOTE
        print("Applying SMOTE to training partition...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"Resampled training partition shape: {X_train_scaled.shape}")
    except ImportError:
        print("imblearn not installed. Skipping SMOTE resampling, fallback will use model parameters.")

    # 4. Serialize split data and scaler
    processed_data_path = os.path.join(data_dir, "processed_data.pkl")
    with open(processed_data_path, "wb") as f:
        pickle.dump({
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler
        }, f)
    print(f"Saved processed splits and scaler to: {processed_data_path}")

if __name__ == "__main__":
    prepare_data()

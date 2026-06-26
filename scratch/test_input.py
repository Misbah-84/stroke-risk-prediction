import pickle
import numpy as np

# Load ensemble model and scaler
with open("models/stroke_ensemble_model.pkl", "rb") as f:
    artifact = pickle.load(f)
ensemble_model = artifact["model"]
scaler = artifact["scaler"]

# Set input parameters
age = 45.0
glucose = 105.5
bmi = 24.5
hypertension = 0
heart_disease = 0

# Scale continuous
cont_raw = np.array([[age, glucose, bmi]])
binary_raw = np.array([[hypertension, heart_disease]])
cont_scaled = scaler.transform(cont_raw)
features_combined = np.hstack((cont_scaled, binary_raw))

# Models
dt_model = ensemble_model.estimators[0]
knn_model = ensemble_model.estimators[1]
svc_model = ensemble_model.estimators[2]

# Predict proba
dt_prob = dt_model.predict_proba(features_combined)[0][1]
knn_prob = knn_model.predict_proba(features_combined)[0][1]
svc_prob = svc_model.predict_proba(features_combined)[0][1]

print(f"DT Prob: {dt_prob}")
print(f"KNN Prob: {knn_prob}")
print(f"SVC Prob: {svc_prob}")
print(f"Ensemble Average: {(dt_prob + knn_prob + svc_prob) / 3.0}")

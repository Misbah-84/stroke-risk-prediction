# StrokeRisk AI: Dual-Engine Clinical Decision Support System

StrokeRisk AI (clinically relevant to localized screening for *Falej*) is an advanced, dual-engine clinical decision-support system. It bridges predictive machine learning workflows with generative artificial intelligence to provide highly accurate stroke risk stratification alongside guideline-grounded medical counseling.

---

## 🚀 System Architecture Overview

The application utilizes a design pattern called a **Context Bridge** to seamlessly link two independent analytical layers:

[Patient Vitals Input]
│
▼
[Data Prep & Scaling] ──► Preserves binary logic, normalizes continuous variables
│
▼
[ML Ensemble Engine]  ──► Soft-Voting Matrix (SVM + Decision Tree + KNN) ──► 65% High Risk
│
▼
[System Prompt Injection]
│
▼
[RAG Engine Chatbot]  ──► Fetches Top 3 Semantic Chunks from Chroma DB (ASA Guidelines)
│
▼
[Gemini 2.5 Flash]    ──► Grounded, clinically sound clinical translation response


1. **The Machine Learning Engine:** Validates incoming user metrics across a custom-built, heterogeneous soft-voting ensemble comprising a Support Vector Classifier (SVC), a K-Nearest Neighbors (KNN) model, and a constrained Decision Tree (DT).
2. **The Retrieval-Augmented Generation (RAG) Engine:** An expert medical chatbot operating at a low temperature ($0.2$) backed by Google's `Gemini 2.5 Flash`. It acts strictly within the analytical boundaries of a local **Chroma Vector Database** containing official American Stroke Association (ASA) and WHO clinical guidelines.

---

## 🛠️ Data Engineering & Pipeline Fixes

### 1. Targeted Variable Scaling
Standard global preprocessing transforms binary features into geometric outliers, severely destroying their boolean structure. StrokeRisk AI isolates continuous features (`age`, `avg_glucose_level`, `bmi`) through a `StandardScaler` while allowing binary flags (`hypertension`, `heart_disease`) to bypass scaling entirely before being horizontally re-stacked with `np.hstack()`. This eliminates training-inference skew and maintains perfect logical interpretability.

### 2. SMOTE Resampling
Medical datasets exhibit intense class imbalances (typically a 95/5 healthy-to-stroke split), introducing the *Accuracy Paradox* where a baseline model scores 95% accuracy while maintaining a 0% minority class recall. By applying **SMOTE (Synthetic Minority Over-sampling Technique)**, the training matrix was expanded from 4,088 samples to 7,778 samples, driving the ensemble's Class 1 recall up to a stable **0.70+**.

### 3. Classification Confidence Refactor
To mirror formal clinical standards, the main dashboard decoupling differentiates *Stroke Risk* from *Classification Confidence*:
* **If Stroke Probability < 50%:** Assigned class is **Healthy**. Confidence is calculated dynamically as $100\% - \text{Stroke Risk}$ (e.g., a 3% risk presents as *97% Confidence: Healthy*).
* **If Stroke Probability ≥ 50%:** Assigned class is **Stroke Risk**. Confidence represents the raw probability vector output directly (e.g., *65% Confidence: Stroke Risk*).

---

## 📁 Repository Structure

```text
├── data/
│   └── healthcare-dataset-stroke-data.csv   # Raw tabular patient dataset
├── documents/
│   ├── Guidelines_Primary_Prevention.pdf     # ASA Reference Text
│   └── stroke_facts.pdf                     # WHO Reference Text
├── models/
│   ├── dt_model.pkl                         # Serialized Decision Tree
│   ├── knn_model.pkl                        # Serialized KNN Weights
│   ├── svc_model.pkl                        # Serialized Support Vector Weights
│   └── stroke_ensemble_model.pkl            # Final pre-fitted soft-voting model
├── data_prep.py                             # Pipeline isolation, scaling, and SMOTE implementation
├── build_rag.py                             # Recursive text chunking, SHA-256 caching, vector database
├── query_rag.py                             # Semantic search matrix orchestration
├── app.py                                   # Streamlit/Flask Web UI Application Gateway
└── README.md                                # Project Documentation



## ⚡ Installation & Execution Guide

### 1. Clone the Space & Initialize Dependencies
```bash
git clone [https://github.com/Misbah-84/stroke-risk-prediction.git](https://github.com/Misbah-84/stroke-risk-prediction.git)
cd stroke-risk-prediction
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

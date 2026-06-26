import os
import pickle
import sys
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Load env file
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    # Propagate GOOGLE_API_KEY if GEMINI_API_KEY is present
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = gemini_key

load_env()

# Import the custom ensemble classifier class definition so pickle can un-serialize it
from ensemble import PreFittedSoftVotingEnsemble

app = Flask(__name__)

# Load models and scaler using pathlib
models_dir = Path(__file__).parent / "models"
model_path = models_dir / "stroke_ensemble_model.pkl"
if not model_path.exists():
    raise FileNotFoundError(f"Ensemble model not found at {model_path}. Please run ensemble.py first.")

with open(model_path, "rb") as f:
    artifact = pickle.load(f)
ensemble_model = artifact["model"]
scaler = artifact["scaler"]

# Safely import LangChain components
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    print(f"[ERROR] LangChain dependencies missing: {e}")
    sys.exit(1)

# Initialize read-only connection to existing local Chroma DB
chroma_db_dir = Path(__file__).parent / "chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(
    persist_directory=str(chroma_db_dir),
    embedding_function=embeddings
)

# Initialize ChatGoogleGenAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

@app.route("/")
def index():
    return render_template("index.html")

def parse_binary_feature(value):
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value > 0 else 0
    
    val_str = str(value).strip().lower()
    if val_str in ("true", "yes", "on", "1", "y"):
        return 1
    if val_str in ("false", "no", "off", "0", "n"):
        return 0
        
    try:
        return 1 if float(val_str) > 0 else 0
    except ValueError:
        pass
        
    return 0

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        # Retrieve form input values with explicit mapping and fallback defaults
        age = float(data.get("age") if data.get("age") is not None else 45.0)
        glucose = float(data.get("glucose") if data.get("glucose") is not None else (data.get("avg_glucose_level") if data.get("avg_glucose_level") is not None else 105.5))
        bmi = float(data.get("bmi") if data.get("bmi") is not None else 24.5)
        
        # Explicitly map incoming binary keys to 1 or 0
        hypertension = parse_binary_feature(data.get("hypertension"))
        heart_disease = parse_binary_feature(data.get("heart_disease"))
        
        # 1. Isolate the continuous variables (age, glucose, bmi) into their own sub-array
        cont_raw = np.array([[age, glucose, bmi]])
        
        # 2. Isolate the binary variables (hypertension, heart_disease) into their own sub-array
        binary_raw = np.array([[hypertension, heart_disease]])
        
        # 3. Apply the unpacked scaler.transform() ONLY to the continuous variable sub-array.
        cont_scaled = scaler.transform(cont_raw)
        
        # 4. Use np.hstack() to horizontally stack the scaled continuous array with the unscaled raw binary integer array (0 or 1), keeping the exact original column index sequence intact.
        features_combined = np.hstack((cont_scaled, binary_raw))
        
        # Inject logging diagnostics before model predictions
        print("\n--- [DIAGNOSTIC LOGGING] ---", flush=True)
        print(f"Raw JSON received:\n{data}", flush=True)
        print(f"Continuous raw sub-array:\n{cont_raw}", flush=True)
        print(f"Continuous scaled sub-array:\n{cont_scaled}", flush=True)
        print(f"Binary raw sub-array:\n{binary_raw}", flush=True)
        print(f"Combined hybrid matrix:\n{features_combined}", flush=True)
        print("----------------------------\n", flush=True)
        
        # Extract individual model estimators predictions
        # Order in ensemble.py: dt, knn, svc
        dt_model = ensemble_model.estimators[0]
        knn_model = ensemble_model.estimators[1]
        svc_model = ensemble_model.estimators[2]
        
        dt_prob = float(dt_model.predict_proba(features_combined)[0][1])
        knn_prob = float(knn_model.predict_proba(features_combined)[0][1])
        svc_prob = float(svc_model.predict_proba(features_combined)[0][1])
        
        # Soft-voting average
        ensemble_prob = (dt_prob + knn_prob + svc_prob) / 3.0
        
        # Classify risk labels matching UI expectations
        risk_label = "LOW"
        if ensemble_prob >= 0.65:
            risk_label = "HIGH"
        elif ensemble_prob >= 0.35:
            risk_label = "MODERATE"
            
        return jsonify({
            "ensemble_prob": round(ensemble_prob, 4),
            "risk_label": risk_label,
            "svm_prob": round(svc_prob, 4),
            "dt_prob": round(dt_prob, 4),
            "knn_prob": round(knn_prob, 4),
            "age": age,
            "glucose": glucose,
            "bmi": bmi,
            "hypertension": hypertension,
            "heart_disease": heart_disease
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    try:
        user_message = data.get("message", "").strip()
        history = data.get("history", [])
        risk_context = data.get("risk_context", None)
        
        if not user_message:
            return jsonify({"reply": "I did not receive a message. Please ask a question."})
            
        # Format the patient diagnostic context summary if available
        patient_context = "No patient diagnostics context loaded yet."
        if risk_context:
            ensemble_prob = risk_context.get('ensemble_prob', 0.0)
            is_healthy = ensemble_prob < 0.50
            target_label = "Healthy" if is_healthy else "Stroke"
            
            ens_conf = (1.0 - ensemble_prob) if is_healthy else ensemble_prob
            svm_conf = (1.0 - risk_context.get('svm_prob', 0.0)) if is_healthy else risk_context.get('svm_prob', 0.0)
            dt_conf = (1.0 - risk_context.get('dt_prob', 0.0)) if is_healthy else risk_context.get('dt_prob', 0.0)
            knn_conf = (1.0 - risk_context.get('knn_prob', 0.0)) if is_healthy else risk_context.get('knn_prob', 0.0)

            patient_context = (
                f"Patient Risk Profile (ML Ensemble):\n"
                f"  Classification       : {risk_context.get('risk_label', 'UNKNOWN')} RISK\n"
                f"  Predicted Class      : {target_label}\n"
                f"  Ensemble Confidence  : {ens_conf * 100:.1f}%\n"
                f"  SVM Confidence       : {svm_conf * 100:.1f}%\n"
                f"  DT Confidence        : {dt_conf * 100:.1f}%\n"
                f"  KNN Confidence       : {knn_conf * 100:.1f}%\n"
                f"  Age                  : {risk_context.get('age', 'N/A')} years\n"
                f"  Average Glucose Level: {risk_context.get('glucose', 'N/A')} mg/dL\n"
                f"  Body Mass Index (BMI): {risk_context.get('bmi', 'N/A')}\n"
                f"  Hypertension         : {'Yes' if risk_context.get('hypertension') else 'No'}\n"
                f"  Heart Disease        : {'Yes' if risk_context.get('heart_disease') else 'No'}"
            )
            
        # Perform similarity search matching query (extracting top 3 medical guidelines chunks)
        import time
        t_rag_start = time.time()
        try:
            docs = db.similarity_search(user_message, k=3)
            guidelines_context = "\n\n".join([
                f"[Source: {Path(d.metadata.get('source', '')).name} Page: {d.metadata.get('page', 0)}]\n{d.page_content}"
                for d in docs
            ])
        except Exception as e:
            print(f"[RAG RETRIEVAL WARNING] Failed to retrieve from Chroma: {e}")
            guidelines_context = "No guidelines could be retrieved from the database."
        t_rag_end = time.time()
        print(f"[TIMING DIAGNOSTIC] Similarity search took {t_rag_end - t_rag_start:.2f} seconds.")
            
        # Build prompt template
        system_instructions = (
            "You are an expert, highly precise Clinical Assistant specializing in stroke prevention.\n"
            "Your role is to translate raw diagnostic numbers into clear clinical insights based strictly on the provided medical guidelines.\n\n"
            
            "CRITICAL ASSIGNMENT INSTRUCTIONS:\n"
            "1. Map the User's Metrics to the Guidelines: Look at the DIAGNOSTIC SYSTEM CONTEXT. Notice numbers like Age, Glucose, and BMI. "
            "Even if the DOCUMENT VECTOR GROUND TRUTHS use broad terms like 'Diabetes Mellitus' or 'Obesity', you must recognize that an "
            "Average Glucose of 220 mg/dL represents severe diabetes/hyperglycemia, a BMI of 35 represents obesity, and the toggles mean chronic history.\n"
            "2. Synthesize the 'Why': Explain how these specific numbers cross the threshold into the independent stroke predictors mentioned in the texts.\n"
            "3. Focus on Actionable Grounded Advice: If the text discusses risk factors (e.g., controlling blood pressure or managing glycemic levels), "
            "extract those focus points as immediate management actions for the user's specific high-scoring metrics.\n"
            "4. Strict Boundary: Do not invent drugs or treatments not found in the text, but do not falsely claim you lack the patient's metrics when they are clearly provided below.\n\n"
            
            "=====================\n"
            "DIAGNOSTIC SYSTEM CONTEXT:\n"
            "{diagnostic_context}\n"
            "=====================\n"
            "DOCUMENT VECTOR GROUND TRUTHS:\n"
            "{vector_context}\n"
            "====================="
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        # Build history messages stack
        messages_history = []
        for h in history:
            role = h.get("role")
            content = h.get("content")
            if role == "user":
                messages_history.append(HumanMessage(content=content))
            elif role == "assistant":
                messages_history.append(AIMessage(content=content))
                
        # Invoke LangChain RAG pipeline
        t_llm_start = time.time()
        chain = prompt_template | llm
        response = chain.invoke({
            "diagnostic_context": patient_context,
            "vector_context": guidelines_context,
            "history": messages_history,
            "question": user_message
        })
        t_llm_end = time.time()
        print(f"[TIMING DIAGNOSTIC] LLM invocation took {t_llm_end - t_llm_start:.2f} seconds.")
        
        return jsonify({"reply": response.content})
    except Exception as e:
        return jsonify({"reply": f"An error occurred in the RAG pipeline: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting StrokeRisk AI Flask Web Server...")
    app.run(host="127.0.0.1", port=5000, debug=True)

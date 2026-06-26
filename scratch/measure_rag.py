import time
import os
import sys

def verify_environment():
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    gemini_key = os.environ.get("GEMINI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key and not google_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

t0 = time.time()
verify_environment()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Test LLM
t_llm_init = time.time()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
print(f"LLM initialized in {time.time() - t_llm_init:.2f}s")

# Test similarity search
t_sim = time.time()
docs = db.similarity_search("warning signs of stroke", k=3)
print(f"Similarity search finished in {time.time() - t_sim:.2f}s")

# Test LLM call
t_llm_call = time.time()
prompt = ChatPromptTemplate.from_template("Answer the question based on guidelines. Guidelines: {context}. Question: {question}")
chain = prompt | llm
response = chain.invoke({
    "context": "\n\n".join([d.page_content for d in docs]),
    "question": "What are the warning signs of a stroke?"
})
print(f"LLM call finished in {time.time() - t_llm_call:.2f}s")
print(f"Response: {response.content[:100]}...")

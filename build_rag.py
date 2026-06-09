import os
import sys
import time

# Safely import LangChain and Google GenAI packages
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from google.genai.errors import ClientError
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
except ImportError:
    # We will handle import failures gracefully in the verify steps
    pass

class RateLimitedEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Custom wrapper around GoogleGenerativeAIEmbeddings to enforce rate limits
    (100 requests/min limit on the Gemini free tier). It batches documents 
    internally and introduces a delay between API requests.
    """
    @staticmethod
    def _prepare_batches(texts: list[str], batch_size: int) -> list[list[str]]:
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        embeddings = []
        # Chunk texts into batches of 20 to ensure we fit in single API payloads and respect quotas
        batches = RateLimitedEmbeddings._prepare_batches(texts, batch_size=20)
        print(f"\n[RATE-LIMITER] Embedding {len(texts)} chunks in {len(batches)} sequential batches...")
        
        for idx, batch in enumerate(batches):
            print(f"[RATE-LIMITER] Processing batch {idx+1}/{len(batches)} ({len(batch)} chunks)...")
            
            success = False
            retries = 5
            backoff = 35
            while not success and retries > 0:
                try:
                    config = self._build_config(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.output_dimensionality,
                    )
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=config,
                    )
                    batch_embeddings = [list(e.values) for e in result.embeddings]
                    embeddings.extend(batch_embeddings)
                    success = True
                except ClientError as e:
                    if e.code == 429:
                        print(f"[RATE-LIMITER] Quota Exceeded (429). Retrying in {backoff} seconds to clear rate limit...")
                        time.sleep(backoff)
                        retries -= 1
                        backoff *= 1.5
                    else:
                        print(f"[RATE-LIMITER] Client error: {e}")
                        raise e
                except Exception as e:
                    print(f"[RATE-LIMITER] Unexpected error: {e}")
                    raise e
            
            if not success:
                raise RuntimeError("Failed to embed batch after maximum retries.")
            
            if idx + 1 < len(batches):
                print("[RATE-LIMITER] Safeguard: Sleeping 15 seconds to respect Gemini API free quota limits...")
                time.sleep(15)
                
        print("[RATE-LIMITER] All embeddings generated successfully.")
        return embeddings

def verify_environment():
    """
    Ensures that the required Google Gemini API key is available in the environment.
    Loads from a local .env file if present.
    """
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
    
    if not gemini_key and not google_key:
        print("[ERROR] Environment variable 'GEMINI_API_KEY' or 'GOOGLE_API_KEY' is missing.")
        print("Please set it in your environment or create a '.env' file before running this script.")
        print("Example (PowerShell): $env:GEMINI_API_KEY='your-api-key'")
        print("Example (Command Prompt): set GEMINI_API_KEY=your-api-key")
        print("Example (Linux/macOS): export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
        
    # If GOOGLE_API_KEY is not set but GEMINI_API_KEY is, propagate it
    if gemini_key and not google_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

def verify_workspace():
    """
    Verifies that the 'documents/' folder exists and prompts the user to add guidelines PDFs.
    """
    cwd = os.getcwd()
    docs_dir = os.path.join(cwd, "documents")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        print(f"[WORKSPACE SETUP] Created missing documents directory at: {docs_dir}")
    
    print("\n=========================================================================")
    print("INSTRUCTION: Please drop your official stroke/medical guidelines PDFs")
    print(f"into the '{docs_dir}' directory.")
    print("=========================================================================\n")
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    return docs_dir, len(pdf_files)

def build_vector_store(docs_dir):
    """
    Ingests PDFs from the documents folder, chunks them, generates embeddings,
    and stores them in a local Chroma vector database.
    """
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
    except ImportError:
        print("[ERROR] Missing required dependencies. Please install them by running:")
        print("  pip install langchain langchain-google-genai pypdf chromadb")
        sys.exit(1)

    print("Step 1: Loading PDF documents from 'documents/' directory...")
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    
    if not documents:
        print("[WARNING] No PDF documents were found or loaded. Database build skipped.")
        print("Please place PDF files in the 'documents/' folder and run the script again.")
        return None

    print(f"Loaded {len(documents)} document pages.")

    print("\nStep 2: Splitting text into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("\nStep 3: Initializing local Chroma database...")
    embeddings = RateLimitedEmbeddings(model="models/gemini-embedding-001")
    chroma_db_dir = os.path.join(os.getcwd(), "chroma_db")
    print(f"Persisting Chroma DB to: {chroma_db_dir}")
    
    db = Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=embeddings
    )
    
    # Generate unique IDs for each chunk based on hash of content & metadata
    import hashlib
    def get_chunk_id(chunk):
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        content = chunk.page_content
        hash_input = f"{source}:{page}:{content}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    print("Checking for existing chunks in the database...")
    chunks_with_ids = [(chunk, get_chunk_id(chunk)) for chunk in chunks]
    all_ids = [cid for _, cid in chunks_with_ids]
    
    # Query all IDs in one batch call to find what already exists
    try:
        res = db.get(ids=all_ids)
        existing_ids = set(res.get("ids", []))
    except Exception as e:
        print(f"[WARNING] Could not check existing IDs from database (likely empty or new DB): {e}")
        existing_ids = set()
        
    new_chunks = []
    new_chunk_ids = []
    for chunk, cid in chunks_with_ids:
        if cid not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(cid)
            
    print(f"Total chunks: {len(chunks)}")
    print(f"Skipped (already ingested): {len(chunks) - len(new_chunks)}")
    print(f"To ingest (new): {len(new_chunks)}")

    if not new_chunks:
        print("All chunks are already ingested. Database is up-to-date!")
        return db

    # Process new chunks in batches and add them to the database
    batch_size = 20
    total_new = len(new_chunks)
    for i in range(0, total_new, batch_size):
        batch_docs = new_chunks[i : i + batch_size]
        batch_ids = new_chunk_ids[i : i + batch_size]
        
        print(f"\n[INGESTION] Processing batch {i // batch_size + 1}/{(total_new - 1) // batch_size + 1} ({len(batch_docs)} chunks)...")
        db.add_documents(documents=batch_docs, ids=batch_ids)
        
        # Compatibility support for older Chroma integrations in LangChain
        if hasattr(db, "persist"):
            db.persist()
            
        if i + batch_size < total_new:
            print("[INGESTION] Safeguard: Sleeping 15 seconds to respect Gemini API free quota limits...")
            time.sleep(15)
            
    print("\nVector database build complete and successfully persisted.")
    return db

def load_chroma_db():
    """
    Operational helper function that loads the existing 'chroma_db/' vector store
    in read-only mode for instant query capabilities without rebuilding.
    """
    cwd = os.getcwd()
    chroma_db_dir = os.path.join(cwd, "chroma_db")
    
    if not os.path.exists(chroma_db_dir):
        print(f"[RECOVERY ERROR] No vector database found at: {chroma_db_dir}")
        return None
        
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
    except ImportError:
        print("[ERROR] Dependencies are missing. Please run:")
        print("  pip install langchain langchain-google-genai pypdf chromadb")
        return None

    print(f"Loading existing vector database from: {chroma_db_dir}")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=embeddings
    )
    print("Vector database loaded successfully.")
    return db

def main():
    verify_environment()
    docs_dir, pdf_count = verify_workspace()
    
    if pdf_count == 0:
        print("[INFO] No PDFs found to process. Ready for PDF injection.")
        print("To install dependencies, run:")
        print("  pip install langchain langchain-google-genai pypdf chromadb")
    else:
        print(f"[INFO] Found {pdf_count} PDF(s) in 'documents/'. Starting RAG pipeline build...")
        build_vector_store(docs_dir)

if __name__ == "__main__":
    main()

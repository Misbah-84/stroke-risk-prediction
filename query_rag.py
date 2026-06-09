import os
import sys
from build_rag import load_chroma_db, verify_environment

def query_database(query_text):
    verify_environment()
    
    db = load_chroma_db()
    if db is None:
        print("[ERROR] Could not load the vector database. Make sure 'chroma_db/' exists and has been built.")
        sys.exit(1)

    print(f"\nQuerying vector database for: '{query_text}'")
    # Retrieve top 3 matching chunks
    results = db.similarity_search_with_score(query_text, k=3)
    
    print("\n--- Top Matching Document Chunks ---")
    for idx, (doc, score) in enumerate(results, start=1):
        print(f"\nResult #{idx} (Similarity Score: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print("-" * 40)
        print(doc.page_content.strip())
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default query for testing
        test_query = "What are the early warning signs of a stroke?"
    else:
        test_query = " ".join(sys.argv[1:])
        
    query_database(test_query)

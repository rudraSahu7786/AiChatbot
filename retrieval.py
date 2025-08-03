# retrieval.py
from ingestion import vector_store

def retrieve_relevant_docs(query, top_k=3):
    """Retrieve relevant documents from vector store"""
    try:
        docs = vector_store.similarity_search(query, k=top_k)
        return docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []
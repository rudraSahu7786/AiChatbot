# ingestion.py
import os
import fitz
import docx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
# Config: Load environment variables
from dotenv import load_dotenv
load_dotenv()
# Config: can also use dotenv
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.environ.get("ASTRA_DB_KEYSPACE")
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION", "documents")  # Fixed: added default value

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=ASTRA_DB_COLLECTION,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE,
)

def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        doc.close()  # Fixed: close document
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return None

def load_text_from_file(file):
    if not file:
        return None
        
    ext = os.path.splitext(file.name)[1].lower()
    file.seek(0)  # Fixed: reset file pointer
    
    if ext == ".pdf":
        return extract_text_from_pdf(file)
    elif ext == ".docx":
        return extract_text_from_docx(file)
    elif ext == ".txt":
        try:
            return file.read().decode("utf-8")
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return None
    else:
        return None

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    if not text or not text.strip():  # Fixed: check for empty text
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.create_documents([text])

def ingest_to_astra(uploaded_files, metadata):
    print("ingest to astra called for : ",uploaded_files)
    all_chunks = []
    for file in uploaded_files:
        text = load_text_from_file(file)
        if not text:
            continue
        chunks = chunk_text(text)
        for chunk in chunks:
            chunk.metadata = {
                "source": file.name,
                **metadata
            }
        all_chunks.extend(chunks)

    if all_chunks:  # Fixed: only add if chunks exist
        try:
            vector_store.add_documents(all_chunks)
        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0
    
    return len(all_chunks)
import os
import time
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import docx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from ingestion import ingest_to_astra, vector_store
from retrieval import retrieve_relevant_docs

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
st.title("🧠 Abhi GPT — Trained on AI concepts")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar content
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose file(s)", accept_multiple_files=True
    )

     # User Inputs
    email = st.text_input("Email ID", placeholder="yourname@example.com")
    product = st.selectbox("Product", options=["RESUME", "NEWS", "RESEARCH_PAPERS","SUPPORT","DOCUMENTATION","CODE","OTHERS"])
    keyword1 = st.text_input("Keyword 1")
    keyword2 = st.text_input("Keyword 2")

    submit = st.button("Submit")

    def is_valid_email(email):
        return re.match(r"[^@]+@[^@]+\.[^@]+", email)
        
    if submit:
        # Input validation
        errors = []
        if not email or not is_valid_email(email):
            errors.append("Please enter a valid Email ID.")
        if not keyword1 or not keyword2:
            errors.append("Please enter both keywords.")
        if not uploaded_files:
            errors.append("Please upload at least one file.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Uploading files..."):
                time.sleep(2)  # Reduced delay
                metadata = {
                    "email": email,
                    "product": product,
                    "keyword1": keyword1,
                    "keyword2": keyword2
                }
                chunk_count = ingest_to_astra(uploaded_files, metadata)
                print(chunk_count)
            st.success(f"All files successfully uploaded for: {email}")
            st.write(f"**Product**: {product}")
            st.write(f"**Keywords**: {keyword1}, {keyword2}")

            for file in uploaded_files:
                st.success(f"Uploaded: `{file.name}`")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask me anything..."):
    # Show spinner while generating
    with st.spinner("Thinking..."):
        try:
            # Save user's message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").markdown(user_input)

            # Retrieve relevant documents
            relevant_docs = retrieve_relevant_docs(user_input)
            
            if relevant_docs:
                # Create RAG prompt with context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                template = """You are a helpful assistant. Use the following context to answer the question:

                Context: {context}

                Question: {question}

                Answer:"""
                
                prompt_template = PromptTemplate.from_template(template)
                formatted_prompt = prompt_template.format(context=context, question=user_input)
            else:
                # Fallback to general prompt
                template = """You are a helpful assistant. Answer the following:

                Question: {question}

                Answer:"""

                prompt_template = PromptTemplate.from_template(template)
                formatted_prompt = prompt_template.format(question=user_input)

            # Generate response using LangChain + Gemini
            response = llm.invoke(formatted_prompt)

            # Show and store assistant response
            st.chat_message("assistant").markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})

        except Exception as e:
            st.error(f"💥 Error: {e}")
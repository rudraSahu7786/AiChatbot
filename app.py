import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load API key from .env
load_dotenv()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Gemini LLM (using Flash-Lite or Gemini-Pro)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",  # or "gemini-1.5-flash", etc. if supported
    temperature=0.3
)

st.title("🧠 Abhi GPT — Powered by Gemini")

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

            # Optional: Create a prompt template
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

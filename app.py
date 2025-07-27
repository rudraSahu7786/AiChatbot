import os
import streamlit as st
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Setting up llm Gemini 
api_key = os.getenv("GEMINI_API_KEY")
st.title("Abhi GPT")
client = genai.Client(api_key=api_key)
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"{prompt}"
    )
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.candidates[0].content.parts[0].text)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.candidates[0].content.parts[0].text})
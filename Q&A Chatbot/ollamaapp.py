import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# LangSmith tracking (optional for Ollama)
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Ollama"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, model_name):
    try:
        llm = Ollama(model=model_name)
        parser = StrOutputParser()
        chain = prompt | llm | parser
        return chain.invoke({"question": question})
    except Exception as e:
        return f"❌ Error: {str(e)}"

# App title
st.title("Enhanced Q&A Chatbot with Ollama")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.title("Settings")
selected_model = st.sidebar.selectbox("Select an AI model:", ["gemma2:2b"])

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []

# Chat area
st.write("### Chat")
for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant", avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(message)

# Chat input
if prompt_input := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append(("user", prompt_input))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_input)

    with st.spinner("Thinking..."):
        response = generate_response(prompt_input, selected_model)

    st.session_state.chat_history.append(("assistant", response))
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)

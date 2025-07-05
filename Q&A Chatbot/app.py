import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

#langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true" 
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot with OpenAI"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question :{question}")
    ]
)

def generate_response(question, api_key, model, temperature, max_tokens):
    try:
        llm = ChatGroq(model=model,
                       api_key=api_key,
                       temperature=temperature,
                       max_tokens=max_tokens)
        parser = StrOutputParser()
        chain = prompt | llm | parser
        return chain.invoke({"question": question})
    except Exception as e:
        return f"❌ Error: {str(e)}"


## title of the app
st.title("Enhanced Q&A Chatbot with ChatGroq")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


## Sidebar for the settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq Api key:",type="password")

#Dropdown
llm = st.sidebar.selectbox("select an Groq AI model:",["gemma2-9b-it","llama-3.3-70b-versatile","llama-3.1-8b-instant"] )

#Adjust response parameter
temperature = st.sidebar.slider("temperature",min_value=0.0,max_value=1.0,value=0.7)

max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

st.write("### Chat")

# Display chat messages from history
for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)

# Input box stays at the bottom
if prompt := st.chat_input("Ask a question..."):
    if not api_key:
        st.warning("Please enter your Groq API key.")
        st.stop()

    # Append user message to history
    st.session_state.chat_history.append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = generate_response(prompt, api_key, llm, temperature, max_tokens)

    # Append assistant response to history
    st.session_state.chat_history.append(("assistant", response))

    with st.chat_message("assistant"):
        st.markdown(response)

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []

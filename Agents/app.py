import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv


# creating the tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)


# Arxiv tool
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search =DuckDuckGoSearchRun(name="Search")

st.title("Langchain chat with search")


## Sidebar for the settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq Api key:",type="password")

#Dropdown
llm = st.sidebar.selectbox("select an Groq AI model:",["gemma2-9b-it","llama-3.3-70b-versatile","llama-3.1-8b-instant"] )

if "messeages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"Assistant",
         "content":"Hi, I am a chatBot who can search how can i help you "
         }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name=llm,streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)


"""
🔄 Section 3: StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

✅ What it does:
This connects LangChain's callback system to Streamlit UI updates in real time.

It captures and displays:

When the LLM is thinking

When it's invoking tools

Intermediate steps (reasoning, thoughts)

🧠 Section 4: search_agent.run(...)
response = search_agent.run(
    st.session_state.messages,
    callbacks=[st_cb]
)
✅ What it does:
Runs the agent with the entire chat history (st.session_state.messages) as input.

Callbacks like st_cb let LangChain print progress and tool usage live in the chat UI.

Returns the final assistant response.





[User Types a Message] ---> [Add to st.session_state.messages]

       |
       v

[initialize_agent(...)]
       |
       v
[AgentType.ZERO_SHOT_REACT_DESCRIPTION]
       |
       v
[LLM decides to use a tool?] ---> YES ---> [Tool is called via callback handler]
       |                                    |
       NO                                   v
       |                         [Tool result passed to LLM]
       v                                    |
[LLM generates final response <-------------|
       |
       v
[st.write(response) inside assistant bubble]
       |
       v
[Append response to st.session_state.messages]


"""
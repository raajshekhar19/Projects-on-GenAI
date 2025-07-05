import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

## load the GROQ API KEY

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)



st.title("Conversation rag with pdf upload and chat history")
st.write("Upload Pdf's and chat with their content...")

api_key = st.text_input("Enter your groq APi key:",type="password")


if api_key:
    llm =ChatGroq(groq_api_key=api_key,model="gemma2-9b-it") 
    
    #chat interface
    session_id = st.text_input("Session ID",value="default_session")

    #satefully manage

    if "store" not in st.session_state:
        st.session_state.store = {}


    #uploading...


    uploaded_files = st.file_uploader("Choose a pdf file :",type="pdf",accept_multiple_files=True)

    #processing upload
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as f:
                f.write(file.getvalue())
                file_name = file.name

            #preparing the docs
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        #split text and create embeddings
        text_splitter =RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=100)
        splited_text = text_splitter.split_documents(documents)
        #creating the embeddings
        vector_store = Chroma.from_documents(documents=splited_text,embedding=embeddings)
        retriever = vector_store.as_retriever()

## rephrazing the question for follow up

        contextulize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "Which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do Not answer the question, "
                "just reformulate it if needed and otherwise return it as it is"
        )

        contextulize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextulize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextulize_q_prompt)
        

        # Answer question prompt
        system_prompt = (
            "You are an assistant for question_answering tasks."
            "Use the following pieces of retrieved context to answer "
            "the question. If you dont't know the answer, say that you "
            "don't know. use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str) ->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]= ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            output_messages_key="answer",
            history_messages_key= "chat_history"
            )
        
        user_input = st.text_input("Your question: ")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )

            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")  # ✅ formatted string
            st.write("Chat History: ",session_history.messages)

"""
┌─────────────────────┐
│ 1. User Uploads PDF │
└─────────┬───────────┘
          ↓
┌────────────────────────────┐
│ Extract & Split Text Chunks│
└─────────┬──────────────────┘
          ↓
┌──────────────────────────────────┐
│ Generate Embeddings (MiniLM)     │
│ Store in Chroma Vector DB        │
└─────────┬────────────────────────┘
          ↓
┌────────────────────────────────────┐
│ User enters query + Session ID     │
└─────────┬──────────────────────────┘
          ↓
┌────────────────────────────────────────────┐
│ Reformulate question (if follow-up)         │
│ Using: ChatHistory + LLM                    │
└─────────┬────────────────────────────────────┘
          ↓
┌──────────────────────────────────┐
│ Retrieve relevant chunks         │
└─────────┬────────────────────────┘
          ↓
┌──────────────────────────────────────┐
│ Prompt LLM with retrieved context     │
└─────────┬────────────────────────────┘
          ↓
┌─────────────────────────────┐
│ Display Answer + Save Chat  │
└─────────────────────────────┘
"""


##ill 2

"""
User uploads PDF(s) ──┐
                     ▼
       LangChain splits text chunks + generates embeddings
                     ▼
           Chroma stores them as vectors
                     ▼
           User inputs a question + session ID
                     ▼
   Reformulator checks for follow-up context (optional)
                     ▼
   Retriever finds top-K chunks matching the question
                     ▼
LLM answers using prompt + context + history (if any)
                     ▼
     Answer is shown, and chat is saved to session

"""

"""
Reformulator checks for follow-up context (optional)

You're telling the LLM:

"Hey, if the current user question depends on earlier conversation,
 rewrite it as a self-contained, standalone question. If it doesn’t,
return it unchanged. Don’t answer it — just rephrase."

 Example:
Chat History:
User: Who is Einstein?
Bot: Albert Einstein was a physicist who developed the theory of relativity.
User: What did he publish in 1905?

Reformulated Question:
👉 "What did Albert Einstein publish in 1905?"

contextulize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextulize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

This creates a structured prompt like:


System: Given a chat history and the latest user question...
Chat History:
    User: Who is Einstein?
    Assistant: Albert Einstein was a physicist...
Human: What did he publish in 1905?


1. ("system", contextulize_q_system_prompt)
This gives global instructions to the LLM. It doesn't change per input. It tells the LLM:

You will receive a chat history and a user question

Your task is not to answer

Your task is to rewrite the question clearly and completely

--------------------------------------------------------------------------

2. MessagesPlaceholder(variable_name="chat_history")
This is key.

It's a placeholder in the prompt structure.

When the app runs, LangChain will insert all messages (previous user+assistant turns) here.

The variable name "chat_history" corresponds to what you're managing inside RunnableWithMessageHistory.

At runtime, this block becomes something like:

User: Who is Einstein?
Assistant: Albert Einstein was a physicist...



----------------------------------------------------------------
so when i ask a question that is based on a past message history the
 llm is called twice once for rephrasing the question and giving it 
 to the reteiever and the second time for actullay answering by getting
   the context from the retriever and answering the rephrased question

   
TWO LLM CALLS IN THE PIPELINE
Here’s what happens when you ask a question like:

??????????????????????????????????????????????????????
“What did he publish in 1905?”
 
1. History-Aware Retriever → Rewriting

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextulize_q_prompt
)

This does:

Looks at:

Chat history (e.g. "Who is Einstein?")

Current input (e.g. "What did he publish in 1905?")

Formats the contextulize_q_prompt using MessagesPlaceholder.

Calls the LLM once to produce:

"What did Albert Einstein publish in 1905?"

✅ This is now a standalone question.


-----------------------------------------------------------------
2: Retriever → Search Vector DB
The rephrased question is passed to the retriever:

retriever.get_relevant_documents(
    "What did Albert Einstein publish in 1905?"
)

It performs a vector search and returns the most relevant chunks.

------------------------------------------------------------------

3: RAG Chain → Answering
second LLM call:
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

This step:

Takes the retrieved chunks

Adds them to the qa_prompt as {context}

Adds the same user input (e.g. "What did he publish in 1905?") or rephrased question

Sends it to the LLM again to generate the final answer


1️⃣ USER INPUT: "What did he publish in 1905?"

│
├──> 🧠 LLM Call #1 → Rewriting Prompt (contextualize_q_prompt)
│     └─ Chat History + Current Input → "What did Albert Einstein publish in 1905?"
│
├──> 🔍 Retriever gets relevant documents from Vector DB using rephrased question
│
└──> 🧠 LLM Call #2 → Answering Prompt (qa_prompt)
      └─ Prompt: "Use the following context to answer the question..."
               + {context}
               + "What did Albert Einstein publish in 1905?"



------------------------------------------------------------------------


🔍 1. create_history_aware_retriever(...)

🔧 Purpose
This function wraps an existing retriever (e.g. Chroma) with an LLM-powered 
question reformulator, enabling the retriever to handle multi-turn conversations.


create_history_aware_retriever(
    llm: BaseLanguageModel,
    base_retriever: BaseRetriever,
    rephrase_prompt: ChatPromptTemplate
)

llm: the LLM used to rephrase the question (e.g., Groq Gemma2-9B).

base_retriever: like Chroma.as_retriever() or FAISS.as_retriever().

rephrase_prompt: defines how to turn the input + history into a standalone question.


🧠 How It Works Internally (Simplified)

class HistoryAwareRetriever(BaseRetriever):

    def __init__(self, llm, retriever, prompt_template):
        ...

    def get_relevant_documents(self, input):
        # Step 1: Extract raw input and chat_history
        query = input["input"]
        history = input["chat_history"]

        # Step 2: Prompt the LLM with history + query
        reformulated_query = llm.invoke(prompt_template.format(...))

        # Step 3: Use base retriever with rewritten query
        return retriever.get_relevant_documents(reformulated_query)

Final Behavior
After you run:
history_aware_retriever = create_history_aware_retriever(...)

You can use it just like any retriever:
docs = history_aware_retriever.get_relevant_documents({
    "input": "What did he publish?",
    "chat_history": chat_messages
})

"""











"""
🧩 2. create_retrieval_chain(...)
🔧 Purpose
This function creates a full RAG (Retrieval-Augmented Generation) Chain, connecting:

Retriever → to fetch documents (history-aware in this case)

StuffDocumentsChain (LLM chain) → to generate answer

It wraps both into a single LangChain Runnable.


create_retrieval_chain(
    retriever: BaseRetriever,
    combine_docs_chain: Runnable
)

retriever: retrieves docs from vector DB (can be history-aware)

combine_docs_chain: LLM chain that takes context + input and generates an answer

🧠 What It Returns
A Runnable chain that:

1.Accepts:
{
  "input": user_question,
  "chat_history": [...],  # Optional
}

2.Calls the retriever:

If it’s history-aware → first reformulates the question

Then retrieves relevant docs

3.Calls the LLM chain:

Formats prompt using retrieved docs + input + history

Sends prompt to LLM


4. returns
{
  "answer": "Here's the final answer",
  "context": [list of Documents],
  ...
}

"""



"""
📊 Visual Flow of Full create_retrieval_chain(...)

                ┌──────────────────────────────┐
Input:          │ input="What did he publish?" │
                │ chat_history=[...]           │
                └─────────────┬────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │ [Retriever: history-aware]                  │
        │  └─ Reformulates question using LLM         │
        │  └─ Retrieves top K docs from vector store  │
        └─────────────┬───────────────────────────────┘
                      ▼
        ┌─────────────────────────────────────┐
        │ [LLM Chain: StuffDocumentsChain]    │
        │  └─ Prompt:                         │
        │     - System: Use context to answer │
        │     - Context: {retrieved chunks}   │
        │     - Question: {input}             │
        └─────────────────────────────────────┘
                      ▼
        ┌────────────────────────────────────┐
        │ Output:                            │
        │  └─ "He published 4 groundbreaking │
        │      papers in 1905..."            │
        └────────────────────────────────────┘

        

🧠 Putting It All Together
Here’s how your app flows:

User question ─────────────────────────────────────┐
                                                  ▼
      ┌──────────────────────────────────────────────┐
      │ create_history_aware_retriever               │
      │   └─ Uses LLM to rewrite question if needed  │
      │   └─ Retrieves relevant docs from Chroma     │
      └──────────────────────────────────────────────┘
                                                  ▼
      ┌──────────────────────────────────────────────┐
      │ create_retrieval_chain                       │
      │   └─ Sends documents + prompt + question     │
      │       to the LLM to answer                   │
      └──────────────────────────────────────────────┘
                                                  ▼
                              Final answer shown to user

                              


"""



"""
you wrap the RAG chain with history management using RunnableWithMessageHistory.

🧩 RunnableWithMessageHistory


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    output_messages_key="answer",  
    history_messages_key="chat_history"
)

🔧 Purpose
This wrapper allows the entire RAG chain to be aware of message history and manage it per session.

It automatically:

  --->Saves user input + model output

  --->Injects chat history into both the rewriting and answering stages

  
🔍 Arguments Breakdown
✅ rag_chain
This is the output of create_retrieval_chain(...). It includes:

The reformulation logic

The retriever

The LLM answering logic

✅ get_session_history
This is your custom function:

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


Stores message history in a per-session dictionary (st.session_state.store)

ChatMessageHistory is a LangChain-provided class that tracks messages like:

[HumanMessage("What is gravity?"), AIMessage("Gravity is...")]


✅ input_messages_key="input"
Tells LangChain where to find the user’s latest message in the input dict:

{
  "input": "What did he publish in 1905?",
  ...
}

✅ output_messages_key="answer"
 This is the key 
LangChain uses to find the model’s output and store it as an AIMessage.

✅ history_messages_key="chat_history"
This tells LangChain what key to use when injecting message history into:

contextulize_q_prompt

qa_prompt

The messages stored in get_session_history() will be injected here using MessagesPlaceholder("chat_history")

⚙️ What It Does Internally
Here’s how RunnableWithMessageHistory modifies your pipeline:


invoke({
  "input": "What did he publish in 1905?"
})

Internally becomes:

# Step 1: Get full history for this session from ChatMessageHistory
chat_history = [
  HumanMessage("Who is Einstein?"),
  AIMessage("Albert Einstein was a physicist...")
]

# Step 2: Insert chat history into:
# - question reformulation prompt
# - question answering prompt

# Step 3: Append this new message to history:
chat_history.append(HumanMessage("What did he publish in 1905?"))
chat_history.append(AIMessage("He published 4 papers in 1905."))  # if answer key is correct

# Step 4: Save back to session_state.store[session_id]


📦 At Runtime, You Call:
response = conversational_rag_chain.invoke(
    {"input": user_input},
    config={"configurable": {"session_id": session_id}}
)
Passes the user's question to the full RAG pipeline

Internally tracks chat history using the session ID


"""
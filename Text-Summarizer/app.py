import validators, streamlit as st
import traceback
from urllib.parse import urlparse, parse_qs

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


# Streamlit UI Setup
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# User Inputs
groq_api_key = st.text_input("Groq API Key", value="", type="password")
generic_url = st.text_input("URL", label_visibility="collapsed")

# Validate API Key
st.write(f"API key provided: {'Yes' if groq_api_key.strip() else 'No'}")
if not groq_api_key.strip():
    st.error("⚠️ Groq API Key is required!")
    st.stop()

# Setup LLM
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Action Button
if st.button("Summarize the Content from YT or Website"):
    if not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or a website.")
    else:
        try:
            with st.spinner("Thinking..."):
                # Handle YouTube vs Website
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        }
                    )

                # Load and summarize
                docs = loader.load()
                if not docs:
                    st.error("⚠️ No content found at the provided URL.")
                    st.stop()

                st.success(" Content loaded. Summarizing...")
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.subheader(" Summary:")
                st.success(output_summary)

        except Exception as e:
            st.error("An error occurred:")
            st.code(traceback.format_exc())

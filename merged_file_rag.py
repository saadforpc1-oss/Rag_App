import os
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG Agent", page_icon="🤖")
st.title("🤖 RAG Agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(BASE_DIR, "documents")
os.makedirs(folder_path, exist_ok=True)

LOADER_MAP = {".pdf": PyPDFLoader, ".txt": TextLoader, ".csv": CSVLoader, ".docx": Docx2txtLoader}

@st.cache_resource(show_spinner="Loading knowledge base...")
def build_chain():
    file_docs = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        loader_cls = LOADER_MAP.get(ext)
        if loader_cls is None:
            continue
        try:
            file_docs.extend(loader_cls(os.path.join(folder_path, filename)).load())
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")

    os.environ.setdefault("USER_AGENT", "rag-agent/1.0")
    urls = ["https://en.wikipedia.org/wiki/Lahore"]
    web_docs = []
    for url in urls:
        try:
            web_docs.extend(WebBaseLoader(url).load())
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")

    docs = file_docs + web_docs
    if not docs:
        st.error(f"No documents loaded. Add files to '{folder_path}' or check URLs.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        st.error("GROQ_API_KEY not set. Add it in Streamlit Cloud secrets.")
        st.stop()
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_key)

    system_prompt = (
        "You are a helpful assistant. Provide answers based on the provided context. "
        "If the information is not in the

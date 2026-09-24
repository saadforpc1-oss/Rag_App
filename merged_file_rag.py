# Required packages — run this to install everything:
# pip install langchain langchain-core langchain-classic langchain-groq langchain-huggingface langchain-text-splitters langchain-community faiss-cpu pypdf sentence-transformers bs4

import os
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

# Step 1a: Load documents from a dedicated "documents" subfolder
# Put your PDFs, TXTs, CSVs, DOCXs inside this subfolder — keep them
# separate from your script files so the loader only ever sees real documents.
BASE_DIR = r"C:\Users\iAT\Desktop\UET COURSE\Rag Agent"
folder_path = os.path.join(BASE_DIR, "documents")

# Create the folder automatically if it doesn't exist yet, so the script
# doesn't crash on a missing directory the first time you run it.
os.makedirs(folder_path, exist_ok=True)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}

file_docs = []
for filename in os.listdir(folder_path):
    ext = os.path.splitext(filename)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        print(f"Skipping unsupported file: {filename}")
        continue
    file_path = os.path.join(folder_path, filename)
    try:
        loader = loader_cls(file_path)
        file_docs.extend(loader.load())
    except Exception as e:
        print(f"Failed to load {filename}: {e}")

print(f"Loaded {len(file_docs)} documents from folder: {folder_path}")

# Step 1b: Load documents from webpage(s)
os.environ.setdefault("USER_AGENT", "rag-agent/1.0")

urls = [
    "https://en.wikipedia.org/wiki/Lahore",   # ← add as many URLs as you want
]

web_docs = []
for url in urls:
    try:
        web_loader = WebBaseLoader(url)
        web_docs.extend(web_loader.load())
    except Exception as e:
        print(f"Failed to load {url}: {e}")

print(f"Loaded {len(web_docs)} documents from web.")

# Step 1c: Combine both sources
docs = file_docs + web_docs
print(f"Loaded {len(docs)} documents total.")

if not docs:
    raise SystemExit(
        f"No documents were loaded. Add files to '{folder_path}' "
        "or check your URL list, then run this script again."
    )

# Step 2: Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Step 3: Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create FAISS vector store and retriever
vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Step 5: Initialize the LLM via Groq
# Set your key as an environment variable instead of hardcoding it:
#   set GROQ_API_KEY=your_new_key_here      (Windows CMD)
#   $env:GROQ_API_KEY="your_new_key_here"   (PowerShell)
groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable before running this script.")
llm = ChatGroq(model="openai/gpt-oss-120b", api_key="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")

# Step 6: Build the RAG chain
system_prompt = (
    "You are a helpful assistant. Provide answers based on the provided context. "
    "If the information is not in the context, use your intelligence to answer the questions."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Step 7: Ask a question
query = input("Enter your query: ")
response = rag_chain.invoke({"input": query})

print("\nQuery:", query)
print("Answer:", response["answer"])
# print("Retrieved Documents:", [doc.page_content for doc in response["context"]])

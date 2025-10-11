import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# Setup Chroma per client

def get_client_vectorstore(client_name):
    return Chroma(
        collection_name=client_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_clients"
    )

# Load documents by file type


def load_and_split(file_path):
    ext = os.path.splitext(file_path)[1].lower()  # get extension
    
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    elif ext== "csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


# Add docs to a client collection


def add_docs_to_client(client_name, file_path, file_type):
    vectorstore = get_client_vectorstore(client_name)
    chunks = load_and_split(file_path, file_type)
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    


# Query per client
def query_client(client_name, question):
    vectorstore = get_client_vectorstore(client_name)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return results

from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

PERSIST_DIR = "./chroma_clients"

def get_client_vectorstore(client_name: str):
    return Chroma(
        collection_name=client_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIR,
    )

def add_docs_for_client(client_name: str, texts: list, metadatas: list = None):
    store = get_client_vectorstore(client_name)
    store.add_texts(texts=texts, metadatas=metadatas)
    return store

def list_clients():
    chroma = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIR
    )
    return [c.name for c in chroma._client.list_collections()]

def delete_client(client_name: str):
    chroma = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIR
    )
    chroma._client.delete_collection(client_name)
    print(f"Deleted client: {client_name}")


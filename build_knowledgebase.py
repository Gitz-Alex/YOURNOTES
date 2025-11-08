import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

load_dotenv()
folder_path = os.getenv("DOCS_PATH")


def load_documents(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        ext = file_name.lower().split(".")[-1]

        if ext == "pdf":
            docs.extend(PyPDFLoader(file_path).load())
        elif ext == "docx":
            docs.extend(Docx2txtLoader(file_path).load())
        elif ext == "txt":
            docs.extend(TextLoader(file_path).load())
        elif ext == "csv":
            text = pd.read_csv(file_path).to_string(index=False)
            docs.append(Document(page_content=text, metadata={"source": file_name}))
        elif ext == "json":
            text = pd.read_json(file_path).to_string(index=False)
            docs.append(Document(page_content=text, metadata={"source": file_name}))

    return docs


# Split text and embed
def build_vectorstore(folder_path=folder_path, save_path="vectorstore"):
    docs = load_documents(folder_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(save_path)
    print(f"Knowledge base built and saved at '{save_path}'.")


def load_vectorstore(save_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    build_vectorstore()

import os
import shutil
from langchain_chroma import Chroma

PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "rag_documents"


def build_vectordb(chunks, embedding_model, persist_dir=PERSIST_DIR):
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )


def load_vectordb(embedding_model, persist_dir=PERSIST_DIR):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )

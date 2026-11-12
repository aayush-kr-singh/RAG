import os
import sys

sys.path.append(os.path.dirname(__file__))

from loader import load_pdfs_from_dir
from chunker import chunk_documents
from embeddings import get_embedding_model
from vectordb import build_vectordb, PERSIST_DIR

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")


def main():
    print(f"Loading PDFs from {DATA_DIR} ...")
    documents = load_pdfs_from_dir(DATA_DIR)
    print(f"Loaded {len(documents)} pages.")

    print("Chunking documents ...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model ...")
    embedding_model = get_embedding_model()

    print(f"Building vector store at {PERSIST_DIR} ...")
    build_vectordb(chunks, embedding_model)
    print("Done. Vector store is ready.")


if __name__ == "__main__":
    main()

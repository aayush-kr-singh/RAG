import os
import pdfplumber
from langchain_core.documents import Document


def load_pdf(pdf_path):

    documents = []
    source = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "page": page_num + 1}
                    )
                )

    return documents


def load_pdfs_from_dir(data_dir):
    """Recursively load every PDF under data_dir into a single list of Documents."""

    documents = []

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                try:
                    documents.extend(load_pdf(pdf_path))
                except Exception as e:
                    print(f"Failed to load {pdf_path}: {e}")

    return documents
import os
import sys

sys.path.append(os.path.dirname(__file__))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from embeddings import get_embedding_model
from vectordb import load_vectordb
from reranker import get_reranker, rerank

OLLAMA_MODEL = "llama3.2"

PROMPT_TEMPLATE = """
Answer using ONLY the provided context.

Rules:
- Give the shortest possible answer that contains the exact fact requested
  (a name, number, date, amount, duration, or short phrase).
- Copy figures, names, and qualifiers (e.g. "Up to", "At least", "Within")
  exactly as written in the context.
- Do not restate the question. Do not add explanations, extra sentences,
  or phrases like "According to the context".
- Do not mention sources or page numbers.
- If the answer is not found, return exactly: I don't know.

Context:
{context}

Question:
{question}

Answer:
"""


class RAGPipeline:
    def __init__(self, top_k=6, retrieve_k=20):
        self.top_k = top_k
        self.retrieve_k = retrieve_k
        self.embedding_model = get_embedding_model()
        self.vectordb = load_vectordb(self.embedding_model)
        self.reranker = get_reranker()
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve(self, query):
        candidates = self.vectordb.similarity_search(query, k=self.retrieve_k)
        return rerank(self.reranker, query, candidates, self.top_k)

    def answer(self, query):
        docs = self.retrieve(query)
        context = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )
        message = self.prompt.format(context=context, question=query)
        response = self.llm.invoke(message)
        return response.content, docs


def main():
    pipeline = RAGPipeline()
    print("RAG ready. Type your question (or 'exit' to quit).")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        answer, docs = pipeline.answer(query)
        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for d in docs:
            print(f" - {d.metadata.get('source')} (page {d.metadata.get('page')})")

def query_rag(question):
    pipeline = RAGPipeline()

    answer, docs = pipeline.answer(question)

    contexts = [doc.page_content for doc in docs]

    return {
        "answer": answer,
        "contexts": contexts
    }
if __name__ == "__main__":
    main()

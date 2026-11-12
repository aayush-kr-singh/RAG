"""
Information Retrieval Metric — RECALL @ K  (Section 4.3)
Computes R@5, R@7, R@10, R@20.

For each question, retrieves top-20 chunks from the vector store.
A question is considered a hit at rank k if any of the top-k chunks
comes from the same PDF as the ground-truth source.

Reads : evaluation/eval_dataset.json  (98 Q)  +  eval_dataset_cleaned.csv  (68 Q)
Saves : evaluation/recall_at_k_results.csv
"""
import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.embeddings import get_embedding_model
from src.vectordb import load_vectordb

JSON_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
CSV_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_dataset_cleaned.csv")
OUTPUT    = os.path.join(os.path.dirname(__file__), "recall_at_k_results.csv")

K_VALUES = [5, 7, 10, 20]
MAX_K    = max(K_VALUES)


def load_combined():
    items = []
    with open(JSON_PATH, encoding="utf-8") as f:
        for d in json.load(f):
            items.append({"question": d["question"], "source": d.get("source", ""), "dataset": "json"})
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        items.append({"question": str(row["question"]), "source": str(row.get("sources", "")), "dataset": "csv"})
    return items


def main():
    data = load_combined()
    print(f"Loaded {len(data)} questions (JSON=98, CSV=68).")

    embedding_model = get_embedding_model()
    vectordb = load_vectordb(embedding_model)
    print("Vector store loaded.\n")

    results = []
    hits = {k: 0 for k in K_VALUES}

    for i, item in enumerate(data):
        question  = item["question"]
        gt_source = item["source"].strip()

        docs = vectordb.similarity_search(question, k=MAX_K)
        retrieved_sources = [d.metadata.get("source", "") for d in docs]

        row_result = {"question": question, "ground_truth_source": gt_source, "dataset": item["dataset"]}
        for k in K_VALUES:
            top_k_sources = retrieved_sources[:k]
            hit = any(gt_source.lower() in s.lower() or s.lower() in gt_source.lower()
                      for s in top_k_sources)
            row_result[f"hit@{k}"] = int(hit)
            if hit:
                hits[k] += 1

        results.append(row_result)
        print(f"[{i+1}/{len(data)}] " +
              " ".join(f"R@{k}={'✓' if row_result[f'hit@{k}'] else '✗'}" for k in K_VALUES))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT, index=False)

    total = len(data)
    print(f"\n{'='*55}")
    print("RECALL @ K  (Information Retrieval)")
    print(f"{'='*55}")
    for k in K_VALUES:
        print(f"R@{k:2d}  :  {hits[k]:3d}/{total}  =  {hits[k]/total*100:.2f}%")
    print(f"{'='*55}")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()

"""
Step 1 — Generate RAG answers for all questions.
Combines both datasets:
  - evaluation/eval_dataset.json  (98 Q, keys: id/question/answer/source/page)
  - eval_dataset_cleaned.csv      (68 Q, columns: question/answer/sources/...)

Total: 166 questions.

Saves: evaluation/generated_answers.csv
Columns: question, generated_answer, ground_truth, context,
         retrieved_sources (pipe-separated filenames), ground_truth_source, dataset
"""
import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.rag import RAGPipeline

JSON_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
CSV_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_dataset_cleaned.csv")
OUTPUT    = os.path.join(os.path.dirname(__file__), "generated_answers.csv")


def load_combined():
    items = []

    # JSON dataset
    with open(JSON_PATH, encoding="utf-8") as f:
        for d in json.load(f):
            items.append({
                "question":            d["question"],
                "ground_truth":        d["answer"],
                "ground_truth_source": d.get("source", ""),
                "dataset":             "json",
            })

    # CSV dataset
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        items.append({
            "question":            str(row["question"]),
            "ground_truth":        str(row["answer"]),
            "ground_truth_source": str(row.get("sources", "")),
            "dataset":             "csv",
        })

    return items


def main():
    data = load_combined()
    print(f"Combined dataset: {len(data)} questions (JSON=98, CSV=68)")

    rag = RAGPipeline()
    print("RAG pipeline ready.\n")

    rows = []
    for i, item in enumerate(data):
        question = item["question"]
        try:
            answer, docs = rag.answer(question)
            context  = "\n\n".join(d.page_content for d in docs)
            ret_srcs = "|".join(d.metadata.get("source", "") for d in docs)
        except Exception as e:
            answer, context, ret_srcs = f"ERROR: {e}", "", ""

        rows.append({
            "question":            question,
            "generated_answer":    answer,
            "ground_truth":        item["ground_truth"],
            "context":             context,
            "retrieved_sources":   ret_srcs,
            "ground_truth_source": item["ground_truth_source"],
            "dataset":             item["dataset"],
        })
        print(f"[{i+1}/{len(data)}] done")

    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(rows)} rows → {OUTPUT}")


if __name__ == "__main__":
    main()

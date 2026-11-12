"""
Evaluator 4 — RETRIEVAL RELEVANCE (Binary)

Goal :
Determine whether the retrieved context is relevant to the user's question.

Mode :
No reference answer required.

Evaluator :
LLM-as-Judge

Score:
    1 = Relevant
    0 = Not Relevant

Reads :
    evaluation/generated_answers.csv

Saves :
    evaluation/retrieval_relevance_scores.csv
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_ollama import ChatOllama



INPUT = os.path.join(os.path.dirname(__file__), "generated_answers.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "retrieval_relevance_scores.csv")



llm = ChatOllama(
    model="llama3.2",
    temperature=0
)



PROMPT = """
You are evaluating the retrieval component of a Retrieval-Augmented Generation (RAG) system.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Determine whether the retrieved context is relevant for answering the question.

SCORING

Return 1 if:
- The retrieved context is relevant to the question.
- It contains the important information needed to answer the question.
- Most of the retrieved content is useful.

Return 0 if:
- The retrieved context is mostly irrelevant.
- It lacks the important information required to answer the question.
- The retrieved documents are unrelated to the user's question.

Be strict.

Return ONLY one character:

1
or
0

Do not explain your decision.
"""



def score(row):

    try:

        response = llm.invoke(
            PROMPT.format(
                question=row["question"],
                context=str(row.get("context", ""))[:3000],
            )
        ).content.strip()

        if response.startswith("1"):
            return 1
        elif response.startswith("0"):
            return 0

    except Exception as e:
        print(f"Error: {e}")

    return 0



def main():

    df = pd.read_csv(INPUT)

    scores = []

    for idx, row in df.iterrows():

        s = score(row)
        scores.append(s)

        print(f"[{idx+1}/{len(df)}] retrieval_relevance = {s}")

    df["retrieval_relevance"] = scores

    df.to_csv(OUTPUT, index=False)

   

    relevant = (df["retrieval_relevance"] == 1).sum()
    not_relevant = (df["retrieval_relevance"] == 0).sum()

    relevance_score = (relevant / len(df)) * 100



    print("\n" + "=" * 60)
    print("BINARY RETRIEVAL RELEVANCE RESULTS")
    print("=" * 60)

    print(f"Total Questions : {len(df)}")
    print(f"Relevant         : {relevant}")
    print(f"Not Relevant     : {not_relevant}")
    print(f"Relevance Score  : {relevance_score:.2f}%")

    print("=" * 60)
    print(f"Saved -> {OUTPUT}")



if __name__ == "__main__":
    main()
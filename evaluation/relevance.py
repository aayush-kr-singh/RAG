"""
Evaluator 2 — RELEVANCE (Binary)

Goal :
Determine whether the generated answer is relevant to the user's question.

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
    evaluation/relevance_scores.csv
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_ollama import ChatOllama



INPUT = os.path.join(os.path.dirname(__file__), "generated_answers.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "relevance_scores.csv")



llm = ChatOllama(
    model="llama3.2",
    temperature=0
)



PROMPT = """
You are evaluating the relevance of a Retrieval-Augmented Generation (RAG) system's answer.

QUESTION:
{question}

GENERATED ANSWER:
{generated}

Determine whether the generated answer is relevant to the user's question.

SCORING

Return 1 if:
- The generated answer directly answers the user's question.
- Most or all of the answer is relevant to the question.
- The answer stays focused on what was asked.

Return 0 if:
- The answer does not answer the question.
- Most of the answer is irrelevant.
- The answer is off-topic.
- The answer is unrelated to the question.

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
                generated=row["generated_answer"],
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

        print(f"[{idx+1}/{len(df)}] relevance = {s}")

    df["relevance"] = scores

    df.to_csv(OUTPUT, index=False)



    relevant = (df["relevance"] == 1).sum()
    not_relevant = (df["relevance"] == 0).sum()

    relevance_score = (relevant / len(df)) * 100

  

    print("\n" + "=" * 55)
    print("BINARY RELEVANCE RESULTS")
    print("=" * 55)

    print(f"Total Questions : {len(df)}")
    print(f"Relevant         : {relevant}")
    print(f"Not Relevant     : {not_relevant}")
    print(f"Relevance Score  : {relevance_score:.2f}%")

    print("=" * 55)
    print(f"Saved -> {OUTPUT}")




if __name__ == "__main__":
    main()
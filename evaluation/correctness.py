"""
Evaluator 1 — CORRECTNESS (Binary)
Goal : Check whether the generated RAG answer is factually correct
       compared to the reference answer.

Mode :
    Requires Ground Truth
    LLM-as-Judge

Score:
    1 = Correct
    0 = Incorrect

Reads :
    evaluation/generated_answers.csv

Saves :
    evaluation/correctness_scores.csv
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_ollama import ChatOllama



INPUT = os.path.join(os.path.dirname(__file__), "generated_answers.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "correctness_scores.csv")



llm = ChatOllama(
    model="llama3.2",
    temperature=0
)



PROMPT = """
Question:
{question}

Correct Answer:
{ground_truth}

Generated Answer:
{generated}

Is the generated answer correct?

Return:
1 = Correct
0 = Incorrect

Different wording is okay if the meaning is the same.

Return only 1 or 0.
"""



def score(row):
    try:
        response = llm.invoke(
            PROMPT.format(
                question=row["question"],
                ground_truth=row["ground_truth"],
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

        print(f"[{idx + 1}/{len(df)}] correctness = {s}")

    df["correctness"] = scores

    df.to_csv(OUTPUT, index=False)


    correct = (df["correctness"] == 1).sum()
    incorrect = (df["correctness"] == 0).sum()

    accuracy = (correct / len(df)) * 100



    print("\n" + "=" * 55)
    print("BINARY CORRECTNESS RESULTS")
    print("=" * 55)

    print(f"Total Questions : {len(df)}")
    print(f"Correct          : {correct}")
    print(f"Incorrect        : {incorrect}")
    print(f"Accuracy         : {accuracy:.2f}%")

    print("=" * 55)
    print(f"Saved -> {OUTPUT}")




if __name__ == "__main__":
    main()
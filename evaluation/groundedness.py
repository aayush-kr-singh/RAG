"""
Evaluator 3 — GROUNDEDNESS (Binary)

Goal :
Determine whether the generated answer is completely supported
by the retrieved context.

Mode :
No reference answer required.

Evaluator :
LLM-as-Judge

Score:
    1 = Grounded
    0 = Not Grounded

Reads :
    evaluation/generated_answers.csv

Saves :
    evaluation/groundedness_scores.csv
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_ollama import ChatOllama



INPUT = os.path.join(os.path.dirname(__file__), "generated_answers.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "groundedness_scores.csv")



llm = ChatOllama(
    model="llama3.2",
    temperature=0
)



PROMPT = """
You are a judge for a Retrieval-Augmented Generation (RAG) system.

Retrieved Context:
{context}

Generated Answer:
{generated}

Task:
Decide whether the generated answer is adequately supported by the retrieved context.

Return 1 if:
- The main answer is supported by the context.
- Minor paraphrasing is acceptable.
- Small wording differences are acceptable.

Return 0 if:
- The main answer is incorrect.
- The answer contradicts the context.
- The answer includes important information that is not supported by the context.
- The answer says "I don't know" even though the context contains the answer.

Return ONLY one digit:
1
or
0
"""



def score(row):

    try:

        response = llm.invoke(
            PROMPT.format(
                context=str(row.get("context", ""))[:3000],
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

        print(f"[{idx+1}/{len(df)}] groundedness = {s}")

    df["groundedness"] = scores

    df.to_csv(OUTPUT, index=False)


    grounded = (df["groundedness"] == 1).sum()
    not_grounded = (df["groundedness"] == 0).sum()

    groundedness_score = (grounded / len(df)) * 100



    print("\n" + "=" * 55)
    print("BINARY GROUNDEDNESS RESULTS")
    print("=" * 55)

    print(f"Total Questions : {len(df)}")
    print(f"Grounded        : {grounded}")
    print(f"Not Grounded    : {not_grounded}")
    print(f"Groundedness    : {groundedness_score:.2f}%")

    print("=" * 55)
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
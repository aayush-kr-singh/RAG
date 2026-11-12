"""
Evaluator 5 — GENERATION QUALITY (Binary)

Evaluates three binary quality dimensions:

1. Specificity
2. Conciseness
3. Completeness

Score:
    1 = Pass
    0 = Fail

Reads :
    evaluation/generated_answers.csv

Saves :
    evaluation/generation_quality_scores.csv
"""

import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_ollama import ChatOllama



INPUT = os.path.join(os.path.dirname(__file__), "generated_answers.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "generation_quality_scores.csv")



llm = ChatOllama(
    model="llama3.2",
    temperature=0
)



PROMPT = """
You are evaluating the quality of a RAG system's generated answer.

QUESTION:
{question}

GENERATED ANSWER:
{generated}

Evaluate the answer on the following three dimensions.

SPECIFICITY

Return 1 if:
- The answer contains precise and specific information.
- Important names, numbers, functions, rules, methods or facts are clearly stated when appropriate.

Return 0 if:
- The answer is vague or generic.
- Important details are missing.

CONCISENESS

Return 1 if:
- The answer directly answers the question.
- There is little or no unnecessary information.

Return 0 if:
- The answer is unnecessarily verbose.
- It contains irrelevant information.

COMPLETENESS

Return 1 if:
- The answer includes all important information needed to answer the question.

Return 0 if:
- Important parts of the answer are missing.
- The answer is incomplete.

Return ONLY in the following format:

Specificity: <0 or 1>
Conciseness: <0 or 1>
Completeness: <0 or 1>

Do not explain your reasoning.
"""



def parse(text):

    spec = 0
    conc = 0
    comp = 0

    for line in text.splitlines():

        line = line.strip().lower()

        if line.startswith("specificity"):
            m = re.search(r"[01]", line)
            if m:
                spec = int(m.group())

        elif line.startswith("conciseness"):
            m = re.search(r"[01]", line)
            if m:
                conc = int(m.group())

        elif line.startswith("completeness"):
            m = re.search(r"[01]", line)
            if m:
                comp = int(m.group())

    return spec, conc, comp



def main():

    df = pd.read_csv(INPUT)

    specs = []
    concs = []
    comps = []

    for idx, row in df.iterrows():

        try:

            response = llm.invoke(
                PROMPT.format(
                    question=row["question"],
                    generated=row["generated_answer"],
                )
            ).content.strip()

            s, c, cp = parse(response)

        except Exception as e:

            print(f"[{idx+1}] Error: {e}")
            s, c, cp = 0, 0, 0

        specs.append(s)
        concs.append(c)
        comps.append(cp)

        print(
            f"[{idx+1}/{len(df)}] "
            f"Specificity={s} "
            f"Conciseness={c} "
            f"Completeness={cp}"
        )

 

    df["specificity"] = specs
    df["conciseness"] = concs
    df["completeness"] = comps

    df.to_csv(OUTPUT, index=False)



    spec_acc = df["specificity"].mean() * 100
    conc_acc = df["conciseness"].mean() * 100
    comp_acc = df["completeness"].mean() * 100

    overall = (
        df["specificity"].sum()
        + df["conciseness"].sum()
        + df["completeness"].sum()
    ) / (len(df) * 3) * 100



    print("\n" + "=" * 60)
    print("BINARY GENERATION QUALITY RESULTS")
    print("=" * 60)

    print(f"Total Questions       : {len(df)}")
    print(f"Specificity           : {spec_acc:.2f}%")
    print(f"Conciseness           : {conc_acc:.2f}%")
    print(f"Completeness          : {comp_acc:.2f}%")
    print(f"Overall Quality Score : {overall:.2f}%")

    print("=" * 60)
    print(f"Saved -> {OUTPUT}")



if __name__ == "__main__":
    main()
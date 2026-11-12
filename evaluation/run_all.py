"""
Master evaluation runner.

Runs the complete evaluation pipeline.

Step 1 : evaluate.py
Step 2 : correctness.py
Step 3 : relevance.py
Step 4 : groundedness.py
Step 5 : retrieval_relevance.py
Step 6 : generation_quality.py
Step 7 : recall_at_k.py

Usage:
    python evaluation/run_all.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


BASE = os.path.dirname(__file__)

# ---------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------

section("STEP 1 / 7 — Generating RAG Answers")

import evaluate
evaluate.main()

# ---------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------

section("STEP 2 / 7 — Correctness")

import correctness
correctness.main()

# ---------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------

section("STEP 3 / 7 — Relevance")

import relevance
relevance.main()

# ---------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------

section("STEP 4 / 7 — Groundedness")

import groundedness
groundedness.main()

# ---------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------

section("STEP 5 / 7 — Retrieval Relevance")

import retrieval_relevance
retrieval_relevance.main()

# ---------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------

section("STEP 6 / 7 — Generation Quality")

import generation_quality
generation_quality.main()

# ---------------------------------------------------------------------
# Step 7
# ---------------------------------------------------------------------

section("STEP 7 / 7 — Recall @ K")

import recall_at_k
recall_at_k.main()

# ---------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------

section("FINAL EVALUATION SUMMARY")

# ---------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------

accuracy = None
correct = 0
incorrect = 0

try:
    corr_df = pd.read_csv(os.path.join(BASE, "correctness_scores.csv"))

    correct = (corr_df["correctness"] == 1).sum()
    incorrect = (corr_df["correctness"] == 0).sum()

    accuracy = (correct / len(corr_df)) * 100

except Exception as e:
    print(f"Could not read correctness_scores.csv : {e}")

# ---------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------

relevance_score = None

try:
    rel_df = pd.read_csv(os.path.join(BASE, "relevance_scores.csv"))

    relevant = (rel_df["relevance"] == 1).sum()
    relevance_score = (relevant / len(rel_df)) * 100

except Exception as e:
    print(f"Could not read relevance_scores.csv : {e}")

# ---------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------

groundedness_score = None

try:
    gnd_df = pd.read_csv(os.path.join(BASE, "groundedness_scores.csv"))

    grounded = (gnd_df["groundedness"] == 1).sum()
    groundedness_score = (grounded / len(gnd_df)) * 100

except Exception as e:
    print(f"Could not read groundedness_scores.csv : {e}")

# ---------------------------------------------------------------------
# Retrieval Relevance
# ---------------------------------------------------------------------

retrieval_score = None

try:
    rr_df = pd.read_csv(os.path.join(BASE, "retrieval_relevance_scores.csv"))

    relevant = (rr_df["retrieval_relevance"] == 1).sum()
    retrieval_score = (relevant / len(rr_df)) * 100

except Exception as e:
    print(f"Could not read retrieval_relevance_scores.csv : {e}")

# ---------------------------------------------------------------------
# Generation Quality
# ---------------------------------------------------------------------

specificity = None
conciseness = None
completeness = None

try:
    gq_df = pd.read_csv(os.path.join(BASE, "generation_quality_scores.csv"))

    specificity = gq_df["specificity"].mean() * 100
    conciseness = gq_df["conciseness"].mean() * 100
    completeness = gq_df["completeness"].mean() * 100

except Exception as e:
    print(f"Could not read generation_quality_scores.csv : {e}")

# ---------------------------------------------------------------------
# Recall @ K
# ---------------------------------------------------------------------

r5 = None
r7 = None
r10 = None
r20 = None

try:
    rk_df = pd.read_csv(os.path.join(BASE, "recall_at_k_results.csv"))

    n = len(rk_df)

    r5 = rk_df["hit@5"].sum() / n * 100
    r7 = rk_df["hit@7"].sum() / n * 100
    r10 = rk_df["hit@10"].sum() / n * 100
    r20 = rk_df["hit@20"].sum() / n * 100

except Exception as e:
    print(f"Could not read recall_at_k_results.csv : {e}")

# ---------------------------------------------------------------------
# Print Summary
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("                 FINAL RAG EVALUATION RESULTS")
print("=" * 70)

print("Dataset                     : JSON (98) + CSV (68) = 166 Questions")

print("\n---------------- RESPONSE GENERATION ----------------")

if accuracy is not None:
    print(f"Correctness                 : {accuracy:.2f}%")
else:
    print("Correctness                 : N/A")

print(f"Correct Answers             : {correct}")
print(f"Incorrect Answers           : {incorrect}")

if relevance_score is not None:
    print(f"Relevance                   : {relevance_score:.2f}%")
else:
    print("Relevance                   : N/A")

if groundedness_score is not None:
    print(f"Groundedness                : {groundedness_score:.2f}%")
else:
    print("Groundedness                : N/A")

if specificity is not None:
    print(f"Specificity                 : {specificity:.2f}%")
else:
    print("Specificity                 : N/A")

if conciseness is not None:
    print(f"Conciseness                 : {conciseness:.2f}%")
else:
    print("Conciseness                 : N/A")

if completeness is not None:
    print(f"Completeness                : {completeness:.2f}%")
else:
    print("Completeness                : N/A")

print("\n-------------------- RETRIEVAL ----------------------")

if retrieval_score is not None:
    print(f"Retrieval Relevance         : {retrieval_score:.2f}%")
else:
    print("Retrieval Relevance         : N/A")

if r5 is not None:
    print(f"Recall@5                   : {r5:.2f}%")
else:
    print("Recall@5                   : N/A")

if r7 is not None:
    print(f"Recall@7                   : {r7:.2f}%")
else:
    print("Recall@7                   : N/A")

if r10 is not None:
    print(f"Recall@10                  : {r10:.2f}%")
else:
    print("Recall@10                  : N/A")

if r20 is not None:
    print(f"Recall@20                  : {r20:.2f}%")
else:
    print("Recall@20                  : N/A")

print("=" * 70)
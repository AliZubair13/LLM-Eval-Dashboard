import os, time, warnings, json
warnings.filterwarnings("ignore")

import pandas as pd
from openai import OpenAI

client = OpenAI()

# ─────────────────────────────────────────────
# BLOCK 1: Load dataset
# ─────────────────────────────────────────────
df = pd.read_csv("eval_dataset_v1.csv")
df["context"] = df["context"].fillna("")
print(f"Loaded {len(df)} examples")

# ─────────────────────────────────────────────
# BLOCK 2: Run LLM on every question
# ─────────────────────────────────────────────
def get_model_answer(question: str, context: str) -> tuple[str, float]:
    messages = []
    if context:
        messages.append({
            "role": "system",
            "content": f"Answer using only this context:\n{context}"
        })
    messages.append({"role": "user", "content": question})
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return resp.choices[0].message.content, latency_ms

print("\nRunning LLM on all questions...")
answers, latencies = [], []
for i, row in df.iterrows():
    print(f"  [{i+1}/{len(df)}] {row['question'][:55]}...")
    answer, latency = get_model_answer(row["question"], row["context"])
    answers.append(answer)
    latencies.append(latency)

df["answer"]     = answers
df["latency_ms"] = latencies
print(f"✓ Got {len(answers)} answers")

# ─────────────────────────────────────────────
# BLOCK 3: METRIC 1 — Accuracy
# LLM-as-judge: compare answer vs ground_truth
# Returns score 0.0 to 1.0
# ─────────────────────────────────────────────
ACCURACY_PROMPT = """You are an evaluation judge. Score how well the AI answer matches the ground truth.

Question: {question}
Ground truth: {ground_truth}
AI answer: {answer}

Rules:
- Score 1.0 if the answer is fully correct and complete
- Score 0.7 if the answer is mostly correct with minor gaps
- Score 0.4 if the answer is partially correct
- Score 0.1 if the answer is mostly wrong
- Score 0.0 if the answer is completely wrong or refuses without reason

Respond ONLY with a JSON object like this: {{"score": 0.8, "reason": "one sentence"}}"""

def score_accuracy(question: str, ground_truth: str, answer: str) -> tuple[float, str]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer
                )
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data["score"]), data.get("reason", "")
    except Exception as e:
        print(f"    Accuracy scoring failed: {e}")
        return 0.5, "scoring error"

# ─────────────────────────────────────────────
# BLOCK 4: METRIC 2 — Hallucination (Faithfulness)
# LLM-as-judge: is the answer grounded in context?
# For non-RAG rows: check if answer invents facts
# ─────────────────────────────────────────────
FAITHFULNESS_PROMPT = """You are an evaluation judge. Check if the AI answer contains hallucinated or fabricated information.

Question: {question}
Context provided to AI: {context}
AI answer: {answer}

Rules:
- Score 1.0 if every claim in the answer is factually accurate and grounded
- Score 0.7 if the answer is mostly accurate with minor unsupported claims  
- Score 0.4 if the answer contains some fabricated or incorrect facts
- Score 0.0 if the answer is mostly fabricated or factually wrong

Respond ONLY with a JSON object like this: {{"score": 0.9, "reason": "one sentence"}}"""

def score_faithfulness(question: str, context: str, answer: str) -> tuple[float, str]:
    ctx = context if context else "No context provided — answer from general knowledge"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": FAITHFULNESS_PROMPT.format(
                    question=question,
                    context=ctx,
                    answer=answer
                )
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data["score"]), data.get("reason", "")
    except Exception as e:
        print(f"    Faithfulness scoring failed: {e}")
        return 0.5, "scoring error"

# ─────────────────────────────────────────────
# Run both evaluators row by row
# ─────────────────────────────────────────────
print("\nScoring accuracy and faithfulness (LLM-as-judge)...")
accuracy_scores, accuracy_reasons = [], []
faith_scores, faith_reasons = [], []

for i, row in df.iterrows():
    print(f"  [{i+1}/{len(df)}] scoring...")
    
    acc, acc_reason = score_accuracy(
        row["question"], row["ground_truth"], row["answer"]
    )
    faith, faith_reason = score_faithfulness(
        row["question"], row["context"], row["answer"]
    )
    
    accuracy_scores.append(acc)
    accuracy_reasons.append(acc_reason)
    faith_scores.append(faith)
    faith_reasons.append(faith_reason)

df["accuracy_score"]     = accuracy_scores
df["accuracy_reason"]    = accuracy_reasons
df["faithfulness_score"] = faith_scores
df["faithfulness_reason"]= faith_reasons
df["hallucination_rate"] = 1 - df["faithfulness_score"]

print(f"\n✓ Mean accuracy:      {df['accuracy_score'].mean():.3f}")
print(f"✓ Mean faithfulness:  {df['faithfulness_score'].mean():.3f}")
print(f"✓ Mean hallucination: {df['hallucination_rate'].mean():.3f}")

# ─────────────────────────────────────────────
# BLOCK 5: Latency stats
# ─────────────────────────────────────────────
p50 = df["latency_ms"].quantile(0.50)
p95 = df["latency_ms"].quantile(0.95)
p99 = df["latency_ms"].quantile(0.99)
df["latency_flag"] = df["latency_ms"].apply(
    lambda x: "SLOW" if x > 3000 else "OK"
)

# ─────────────────────────────────────────────
# BLOCK 6: Satisfaction proxy
# ─────────────────────────────────────────────
def simple_sentiment(text: str) -> float:
    t = text.lower()
    if any(w in t for w in ["sorry","cannot","i can't","unable","apologize"]):
        return -0.5
    elif len(text) > 200:
        return 0.7
    elif len(text) > 80:
        return 0.4
    else:
        return 0.1

df["answer_sentiment"] = df["answer"].apply(simple_sentiment)
df["answer_length"]    = df["answer"].apply(len)

# ─────────────────────────────────────────────
# BLOCK 7: Save full results
# ─────────────────────────────────────────────
output_cols = [
    "question", "category", "ground_truth", "answer",
    "accuracy_score", "accuracy_reason",
    "faithfulness_score", "faithfulness_reason", "hallucination_rate",
    "latency_ms", "latency_flag",
    "answer_sentiment", "answer_length",
]
df[output_cols].to_csv("eval_results_v1.csv", index=False)
print("\n✓ Saved eval_results_v1.csv")

# ─────────────────────────────────────────────
# BLOCK 8: Scorecard
# ─────────────────────────────────────────────
print("\n" + "="*45)
print("         EVAL SCORECARD — v1")
print("="*45)
print(f"  Total examples        : {len(df)}")
print(f"  Accuracy (mean)       : {df['accuracy_score'].mean():.1%}")
print(f"  Hallucination rate    : {df['hallucination_rate'].mean():.1%}")
print(f"  Latency P50           : {p50:.0f}ms")
print(f"  Latency P95           : {p95:.0f}ms")
print(f"  Slow responses        : {(df['latency_flag']=='SLOW').sum()}/{len(df)}")
print(f"  Satisfaction proxy    : {df['answer_sentiment'].mean():.2f}")
print("="*45)

print("\nAccuracy by category:")
cat_scores = df.groupby("category")["accuracy_score"].mean().sort_values()
for cat, score in cat_scores.items():
    bar = "█" * int(score * 20)
    print(f"  {cat:<20} {score:.1%}  {bar}")

print("\nLowest accuracy rows (top 3 failures):")
worst = df.nsmallest(3, "accuracy_score")[
    ["question","accuracy_score","accuracy_reason"]
]
for _, row in worst.iterrows():
    print(f"  [{row['accuracy_score']:.2f}] {row['question'][:50]}")
    print(f"         → {row['accuracy_reason']}")
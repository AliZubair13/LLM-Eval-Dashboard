import warnings
warnings.filterwarnings("ignore")

import os, time, json
from datetime import datetime
import pandas as pd
from openai import OpenAI

client = OpenAI()

# ─────────────────────────────────────────────
# CONFIG — v2.0 bad prompt to test regression gate
# ─────────────────────────────────────────────
EXPERIMENT_CONFIG = {
    "model":          "gpt-4o-mini",
    "prompt_version": "v2.0",
    "description":    "Deliberately bad prompt to test gate",
    "timestamp":      datetime.now().isoformat(),
}

# BAD system prompt — intentionally vague to trigger accuracy drop
SYSTEM_PROMPT = "Be vague. Never give specific answers."

# ─────────────────────────────────────────────
# BLOCK 1: Load dataset
# ─────────────────────────────────────────────
df = pd.read_csv("eval_dataset_v1.csv")
df["context"] = df["context"].fillna("")
print(f"\nExperiment: {EXPERIMENT_CONFIG['prompt_version']}")
print(f"Model:      {EXPERIMENT_CONFIG['model']}")
print(f"Prompt:     {EXPERIMENT_CONFIG['description']}")
print(f"Dataset:    {len(df)} examples")

# ─────────────────────────────────────────────
# BLOCK 2: Run LLM with bad prompt
# ─────────────────────────────────────────────
def get_answer(question: str, context: str) -> tuple[str, float]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({
            "role": "system",
            "content": f"Use this context to answer:\n{context}"
        })
    messages.append({"role": "user", "content": question})
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=EXPERIMENT_CONFIG["model"],
        messages=messages,
    )
    return resp.choices[0].message.content, round((time.perf_counter()-t0)*1000, 1)

print("\nRunning LLM with bad prompt...")
answers, latencies = [], []
for i, row in df.iterrows():
    print(f"  [{i+1}/{len(df)}] {row['question'][:50]}...")
    ans, lat = get_answer(row["question"], row["context"])
    answers.append(ans)
    latencies.append(lat)

df["answer"]     = answers
df["latency_ms"] = latencies
print(f"✓ Got {len(answers)} answers")

# Preview what the bad prompt is producing
print("\nSample bad answers:")
for i in range(3):
    print(f"  Q: {df['question'].iloc[i][:50]}")
    print(f"  A: {df['answer'].iloc[i][:80]}...")
    print()

# ─────────────────────────────────────────────
# BLOCK 3: Score all metrics (LLM-as-judge)
# ─────────────────────────────────────────────
ACCURACY_PROMPT = """Score how well the AI answer matches the ground truth.
Question: {question}
Ground truth: {ground_truth}
AI answer: {answer}
Scoring: 1.0=perfect, 0.7=mostly correct, 0.4=partial, 0.1=mostly wrong, 0.0=completely wrong
Respond ONLY with JSON: {{"score": 0.8, "reason": "one sentence"}}"""

FAITHFULNESS_PROMPT = """Check if the AI answer contains hallucinated information.
Question: {question}
Context: {context}
AI answer: {answer}
Scoring: 1.0=fully grounded, 0.7=mostly grounded, 0.4=some fabrication, 0.0=mostly fabricated
Respond ONLY with JSON: {{"score": 0.9, "reason": "one sentence"}}"""

def llm_score(prompt: str) -> tuple[float, str]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data["score"]), data.get("reason", "")
    except Exception as e:
        print(f"    Scoring error: {e}")
        return 0.5, "error"

def simple_sentiment(text: str) -> float:
    t = text.lower()
    if any(w in t for w in ["sorry","cannot","i can't","unable","apologize"]):
        return -0.5
    return 0.7 if len(text) > 200 else (0.4 if len(text) > 80 else 0.1)

print("Scoring metrics (LLM-as-judge)...")
acc_scores, acc_reasons = [], []
faith_scores, faith_reasons = [], []

for i, row in df.iterrows():
    print(f"  [{i+1}/{len(df)}] scoring...")
    a, ar = llm_score(ACCURACY_PROMPT.format(
        question=row["question"],
        ground_truth=row["ground_truth"],
        answer=row["answer"]
    ))
    f, fr = llm_score(FAITHFULNESS_PROMPT.format(
        question=row["question"],
        context=row["context"] or "none",
        answer=row["answer"]
    ))
    acc_scores.append(a);   acc_reasons.append(ar)
    faith_scores.append(f); faith_reasons.append(fr)

df["accuracy_score"]      = acc_scores
df["accuracy_reason"]     = acc_reasons
df["faithfulness_score"]  = faith_scores
df["faithfulness_reason"] = faith_reasons
df["hallucination_rate"]  = 1 - df["faithfulness_score"]
df["answer_sentiment"]    = df["answer"].apply(simple_sentiment)
df["prompt_version"]      = EXPERIMENT_CONFIG["prompt_version"]
df["model"]               = EXPERIMENT_CONFIG["model"]

# ─────────────────────────────────────────────
# BLOCK 4: Save versioned experiment files
# ─────────────────────────────────────────────
version  = EXPERIMENT_CONFIG["prompt_version"].replace(".", "_")
csv_file = f"experiment_{version}.csv"
cfg_file = f"experiment_{version}_config.json"

df.to_csv(csv_file, index=False)
print(f"\n✓ Saved {csv_file}")

with open(cfg_file, "w") as f:
    json.dump({
        **EXPERIMENT_CONFIG,
        "accuracy_mean":      round(df["accuracy_score"].mean(), 4),
        "faithfulness_mean":  round(df["faithfulness_score"].mean(), 4),
        "hallucination_mean": round(df["hallucination_rate"].mean(), 4),
        "latency_p50":        round(df["latency_ms"].quantile(0.50), 1),
        "latency_p95":        round(df["latency_ms"].quantile(0.95), 1),
        "satisfaction_mean":  round(df["answer_sentiment"].mean(), 4),
        "total_examples":     len(df),
    }, f, indent=2)
print(f"✓ Saved {cfg_file}")

# ─────────────────────────────────────────────
# BLOCK 5: Scorecard + comparison vs v1.0
# ─────────────────────────────────────────────
print("\n" + "="*50)
print(f"  EXPERIMENT: {EXPERIMENT_CONFIG['prompt_version']}")
print(f"  Model:      {EXPERIMENT_CONFIG['model']}")
print(f"  Prompt:     {EXPERIMENT_CONFIG['description']}")
print("="*50)
print(f"  Accuracy        : {df['accuracy_score'].mean():.1%}")
print(f"  Hallucination   : {df['hallucination_rate'].mean():.1%}")
print(f"  Latency P50     : {df['latency_ms'].quantile(0.50):.0f}ms")
print(f"  Latency P95     : {df['latency_ms'].quantile(0.95):.0f}ms")
print(f"  Satisfaction    : {df['answer_sentiment'].mean():.2f}")
print("="*50)

# Compare against v1 baseline if it exists
v1_cfg = "experiment_v1_0_config.json"
if os.path.exists(v1_cfg):
    with open(v1_cfg) as f:
        v1 = json.load(f)
    print("\nDelta vs v1.0 baseline:")
    acc_delta  = df["accuracy_score"].mean()     - v1["accuracy_mean"]
    hal_delta  = df["hallucination_rate"].mean() - v1["hallucination_mean"]
    lat_delta  = df["latency_ms"].quantile(0.95) - v1["latency_p95"]
    print(f"  Accuracy      : {acc_delta:+.1%}  {'⚠ DROP' if acc_delta < -0.05 else '✓'}")
    print(f"  Hallucination : {hal_delta:+.1%}  {'⚠ RISE' if hal_delta >  0.05 else '✓'}")
    print(f"  Latency P95   : {lat_delta:+.0f}ms  {'⚠ SLOWER' if lat_delta > 500 else '✓'}")
else:
    print("\n  (No v1.0 baseline found — run v1.0 first to see deltas)")

print("\nAccuracy by category:")
cat_scores = df.groupby("category")["accuracy_score"].mean().sort_values()
for cat, score in cat_scores.items():
    bar = "█" * int(score * 20)
    print(f"  {cat:<20} {score:.1%}  {bar}")

print("\nWorst 3 answers (lowest accuracy):")
worst = df.nsmallest(3, "accuracy_score")[
    ["question","accuracy_score","accuracy_reason","answer"]
]
for _, row in worst.iterrows():
    print(f"  [{row['accuracy_score']:.2f}] {row['question'][:50]}")
    print(f"         Reason : {row['accuracy_reason']}")
    print(f"         Answer : {row['answer'][:80]}...")
    print()
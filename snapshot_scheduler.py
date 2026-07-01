import warnings
warnings.filterwarnings("ignore")

import json, time, schedule
from datetime import datetime
import pandas as pd
from openai import OpenAI

client = OpenAI()

SNAPSHOT_LOG = "snapshots.jsonl"

def take_snapshot():
    """
    Runs a mini eval on 5 sample questions every hour.
    Each run appends one snapshot to snapshots.jsonl.
    In production: replace with your real traffic sample.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Taking snapshot...")

    sample_questions = [
        ("What is the boiling point of water?",
         "100 degrees Celsius at standard pressure."),
        ("Who painted the Mona Lisa?",
         "Leonardo da Vinci."),
        ("How many planets are in the solar system?",
         "Eight planets."),
        ("What is the capital of France?",
         "Paris."),
        ("Who wrote Romeo and Juliet?",
         "William Shakespeare."),
    ]

    JUDGE_PROMPT = """Score this answer vs ground truth. 
Question: {q}
Ground truth: {gt}
Answer: {a}
Respond ONLY with JSON: {{"score": 0.9}}"""

    scores = []
    latencies = []

    for question, ground_truth in sample_questions:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        answer = resp.choices[0].message.content

        # Score it
        judge = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                q=question, gt=ground_truth, a=answer
            )}],
            response_format={"type": "json_object"},
        )
        score = json.loads(judge.choices[0].message.content).get("score", 0.5)
        scores.append(score)
        latencies.append(latency_ms)

    # Save snapshot
    snapshot = {
        "timestamp":      datetime.now().isoformat(),
        "accuracy_mean":  round(sum(scores) / len(scores), 4),
        "latency_p95":    round(sorted(latencies)[int(len(latencies)*0.95)], 1),
        "latency_p50":    round(sorted(latencies)[len(latencies)//2], 1),
        "sample_size":    len(sample_questions),
    }
    with open(SNAPSHOT_LOG, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    print(f"  Accuracy:    {snapshot['accuracy_mean']:.1%}")
    print(f"  Latency P95: {snapshot['latency_p95']:.0f}ms")
    print(f"✓ Snapshot saved to {SNAPSHOT_LOG}")

# ── Run once immediately then every hour ──
print("Snapshot scheduler started.")
print("Running first snapshot now...")
take_snapshot()

print("\nScheduled to run every 60 minutes.")
print("Press Ctrl+C to stop.\n")
schedule.every(60).minutes.do(take_snapshot)

while True:
    schedule.run_pending()
    time.sleep(30)
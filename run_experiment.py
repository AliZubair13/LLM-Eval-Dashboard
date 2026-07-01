import warnings
warnings.filterwarnings("ignore")

import os, time, json, glob
from datetime import datetime
import pandas as pd
from openai import OpenAI
import anthropic
from groq import Groq

# ── Clients ──
openai_client    = OpenAI()
anthropic_client = anthropic.Anthropic()
groq_client      = Groq()

# ─────────────────────────────────────────────
# MODEL REGISTRY — all 4 models
# ─────────────────────────────────────────────
MODELS_TO_TEST = [
    {
        "model":                  "gpt-4o-mini",
        "provider":               "openai",
        "display_name":           "GPT-4o Mini",
        "description":            "Fast cheap OpenAI model",
        "prompt_version":         "v1.0",
        "cost_per_1k_prompt":     0.00015,
        "cost_per_1k_completion": 0.00060,
    },
    {
        "model":                  "gpt-4o",
        "provider":               "openai",
        "display_name":           "GPT-4o",
        "description":            "Best OpenAI model",
        "prompt_version":         "v1.0",
        "cost_per_1k_prompt":     0.0025,
        "cost_per_1k_completion": 0.01,
    },
    {
        "model":                  "claude-sonnet-4-5",
        "provider":               "anthropic",
        "display_name":           "Claude Sonnet",
        "description":            "Anthropic flagship model",
        "prompt_version":         "v1.0",
        "cost_per_1k_prompt":     0.003,
        "cost_per_1k_completion": 0.015,
    },
    {
        "model":                  "llama-3.1-8b-instant",
        "provider":               "groq",
        "display_name":           "Llama 3.1 8B",
        "description":            "Meta Llama 3.1 via Groq (free)",
        "prompt_version":         "v1.0",
        "cost_per_1k_prompt":     0.00005,
        "cost_per_1k_completion": 0.00008,
    },
]

SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."

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

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def call_model(model_cfg: dict, question: str,
               context: str) -> tuple[str, float, int, int]:
    provider = model_cfg["provider"]
    model    = model_cfg["model"]
    t0       = time.perf_counter()

    if provider == "openai":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            messages.append({
                "role": "system",
                "content": f"Use this context:\n{context}"
            })
        messages.append({"role": "user", "content": question})
        resp = openai_client.chat.completions.create(
            model=model, messages=messages,
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return (
            resp.choices[0].message.content,
            latency,
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )

    elif provider == "anthropic":
        system = SYSTEM_PROMPT
        if context:
            system += f"\n\nUse this context:\n{context}"
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return (
            resp.content[0].text,
            latency,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
        )

    elif provider == "groq":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            messages.append({
                "role": "system",
                "content": f"Use this context:\n{context}"
            })
        messages.append({"role": "user", "content": question})
        resp = groq_client.chat.completions.create(
            model=model, messages=messages,
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return (
            resp.choices[0].message.content,
            latency,
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def llm_score(prompt: str) -> tuple[float, str]:
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return float(data["score"]), data.get("reason", "")
    except Exception as e:
        return 0.5, f"error: {e}"


def simple_sentiment(text: str) -> float:
    t = text.lower()
    if any(w in t for w in ["sorry", "cannot", "i can't",
                             "unable", "apologize"]):
        return -0.5
    return 0.7 if len(text) > 200 else (0.4 if len(text) > 80 else 0.1)


def calc_cost(pt: int, ct: int, cfg: dict) -> float:
    return round(
        (pt / 1000 * cfg["cost_per_1k_prompt"]) +
        (ct / 1000 * cfg["cost_per_1k_completion"]),
        6
    )


def composite(cfg: dict) -> float:
    return (
        cfg["accuracy_mean"]                            * 0.40 +
        (1 - cfg["hallucination_mean"])                 * 0.30 +
        (1 - min(cfg["latency_p95"] / 5000, 1))        * 0.15 +
        (1 - min(cfg["cost_per_query_avg"] / 0.01, 1)) * 0.15
    )

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
df_base = pd.read_csv("eval_dataset_v1.csv")
df_base["context"] = df_base["context"].fillna("")
print(f"Dataset: {len(df_base)} examples")
print(f"Models to run: {len(MODELS_TO_TEST)}")
print(f"Estimated time: ~{len(MODELS_TO_TEST) * 3} minutes\n")

all_configs = []

# ─────────────────────────────────────────────
# RUN ALL MODELS
# ─────────────────────────────────────────────
for model_idx, model_cfg in enumerate(MODELS_TO_TEST):
    display  = model_cfg["display_name"]
    model    = model_cfg["model"]
    provider = model_cfg["provider"]

    print(f"\n{'='*55}")
    print(f"  [{model_idx+1}/{len(MODELS_TO_TEST)}] Model    : {display}")
    print(f"  Provider : {provider}")
    print(f"  API ID   : {model}")
    print(f"{'='*55}")

    df     = df_base.copy()
    errors = 0
    answers, latencies = [], []
    pt_list, ct_list   = [], []

    # ── Run LLM ──
    print("  Calling model...")
    for i, row in df.iterrows():
        print(f"    [{i+1}/{len(df)}] {row['question'][:45]}...")
        try:
            ans, lat, pt, ct = call_model(
                model_cfg, row["question"], row["context"]
            )
        except Exception as e:
            print(f"    ⚠ API error: {e}")
            ans, lat, pt, ct = f"ERROR: {e}", 0, 0, 0
            errors += 1
        answers.append(ans)
        latencies.append(lat)
        pt_list.append(pt)
        ct_list.append(ct)

    df["answer"]            = answers
    df["latency_ms"]        = latencies
    df["prompt_tokens"]     = pt_list
    df["completion_tokens"] = ct_list
    df["cost_per_query"]    = [
        calc_cost(pt, ct, model_cfg)
        for pt, ct in zip(pt_list, ct_list)
    ]

    # ── Score metrics ──
    print("\n  Scoring metrics (LLM-as-judge)...")
    acc_scores,   acc_reasons   = [], []
    faith_scores, faith_reasons = [], []

    for i, row in df.iterrows():
        print(f"    [{i+1}/{len(df)}] scoring...")
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
        acc_scores.append(a);    acc_reasons.append(ar)
        faith_scores.append(f);  faith_reasons.append(fr)

    df["accuracy_score"]      = acc_scores
    df["accuracy_reason"]     = acc_reasons
    df["faithfulness_score"]  = faith_scores
    df["faithfulness_reason"] = faith_reasons
    df["hallucination_rate"]  = 1 - df["faithfulness_score"]
    df["answer_sentiment"]    = df["answer"].apply(simple_sentiment)
    df["model"]               = model
    df["display_name"]        = display
    df["prompt_version"]      = model_cfg["prompt_version"]
    df["latency_flag"]        = df["latency_ms"].apply(
        lambda x: "SLOW" if x > 3000 else "OK"
    )

    # ── Save CSV ──
    safe  = display.lower().replace(" ", "_").replace("-", "_")
    csv_f = f"experiment_{safe}.csv"
    cfg_f = f"experiment_{safe}_config.json"
    df.to_csv(csv_f, index=False)
    print(f"\n  ✓ Saved {csv_f}")

    # ── Build config ──
    config = {
        **model_cfg,
        "display_name":          display,
        "safe_name":             safe,
        "timestamp":             datetime.now().isoformat(),
        "accuracy_mean":         round(df["accuracy_score"].mean(), 4),
        "faithfulness_mean":     round(df["faithfulness_score"].mean(), 4),
        "hallucination_mean":    round(df["hallucination_rate"].mean(), 4),
        "latency_p50":           round(df["latency_ms"].quantile(0.50), 1),
        "latency_p95":           round(df["latency_ms"].quantile(0.95), 1),
        "satisfaction_mean":     round(df["answer_sentiment"].mean(), 4),
        "total_examples":        len(df),
        "total_cost_usd":        round(df["cost_per_query"].sum(), 6),
        "cost_per_query_avg":    round(df["cost_per_query"].mean(), 6),
        "prompt_tokens_avg":     round(df["prompt_tokens"].mean(), 1),
        "completion_tokens_avg": round(df["completion_tokens"].mean(), 1),
        "slow_responses":        int((df["latency_flag"] == "SLOW").sum()),
        "errors":                errors,
    }
    with open(cfg_f, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved {cfg_f}")
    all_configs.append(config)

    # ── Per-model scorecard ──
    print(f"\n  ── SCORECARD: {display} ──")
    print(f"  Accuracy       : {config['accuracy_mean']:.1%}")
    print(f"  Hallucination  : {config['hallucination_mean']:.1%}")
    print(f"  Latency P95    : {config['latency_p95']:.0f}ms")
    print(f"  Cost/query     : ${config['cost_per_query_avg']:.6f}")
    print(f"  Total run cost : ${config['total_cost_usd']:.4f}")
    if errors:
        print(f"  ⚠ Errors       : {errors}/{len(df)}")

# ─────────────────────────────────────────────
# SAVE COMBINED SUMMARY
# ─────────────────────────────────────────────
summary_df = pd.DataFrame(all_configs)
summary_df.to_csv("model_comparison_summary.csv", index=False)
print(f"\n✓ Saved model_comparison_summary.csv")

# ─────────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────────
best_name = max(all_configs, key=composite)["display_name"]

print(f"\n{'='*72}")
print("  FINAL MULTI-MODEL COMPARISON")
print(f"{'='*72}")
print(f"  {'Model':<16} {'Accuracy':>9} {'Halluc':>8} "
      f"{'P95ms':>7} {'Cost/q':>12} {'Composite':>10}")
print(f"  {'-'*66}")

for cfg in sorted(all_configs, key=composite, reverse=True):
    tag = " ⭐" if cfg["display_name"] == best_name else ""
    print(
        f"  {cfg['display_name']:<16} "
        f"{cfg['accuracy_mean']:>8.1%} "
        f"{cfg['hallucination_mean']:>7.1%} "
        f"{cfg['latency_p95']:>6.0f} "
        f"${cfg['cost_per_query_avg']:>10.6f} "
        f"{composite(cfg):>9.1%}{tag}"
    )

print(f"{'='*72}")
print(f"\n  ⭐ Best overall model: {best_name}")
print(f"  Weighted: accuracy 40%, faithfulness 30%, "
      f"speed 15%, cost 15%")
print(f"\n  Next step: python -m streamlit run dashboard.py")
import warnings
warnings.filterwarnings("ignore")

import json, sys
import pandas as pd

# ─────────────────────────────────────────────
# THRESHOLDS — tune these for your use case
# ─────────────────────────────────────────────
THRESHOLDS = {
    "accuracy_drop_max":      0.05,   # block if accuracy drops more than 5%
    "hallucination_rise_max": 0.05,   # block if hallucination rises more than 5%
    "latency_p95_rise_max":   500,    # block if P95 rises more than 500ms
    "min_accuracy":           0.80,   # block if accuracy falls below 80% absolute
    "max_hallucination":      0.20,   # block if hallucination exceeds 20% absolute
}

# ─────────────────────────────────────────────
# Load two experiment configs to compare
# ─────────────────────────────────────────────
# Baseline = previous experiment (what's in prod)
# Candidate = new experiment (what you want to deploy)

BASELINE_CONFIG  = "experiment_v1_0_config.json"
CANDIDATE_CONFIG = "experiment_v2_0_config.json" # change to v2_0 when you have one

print("="*55)
print("  REGRESSION GATE")
print("="*55)

try:
    with open(BASELINE_CONFIG)  as f: baseline  = json.load(f)
    with open(CANDIDATE_CONFIG) as f: candidate = json.load(f)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Run run_experiment.py first to generate config files.")
    sys.exit(1)

print(f"\n  Baseline  : {baseline['prompt_version']}  ({baseline['model']})")
print(f"  Candidate : {candidate['prompt_version']} ({candidate['model']})")

# ─────────────────────────────────────────────
# Compare metrics
# ─────────────────────────────────────────────
checks = []

def check(name, baseline_val, candidate_val, threshold,
          direction="lower_is_worse", unit=""):
    if direction == "lower_is_worse":
        delta = candidate_val - baseline_val
        failed = delta < -threshold
        symbol = "▼" if delta < 0 else "▲"
    else:
        delta = candidate_val - baseline_val
        failed = delta > threshold
        symbol = "▲" if delta > 0 else "▼"

    status = "❌ FAIL" if failed else "✅ PASS"
    checks.append(failed)
    print(f"\n  {status} {name}")
    print(f"         Baseline:  {baseline_val:.4f}{unit}")
    print(f"         Candidate: {candidate_val:.4f}{unit}")
    print(f"         Delta:     {symbol}{abs(delta):.4f}{unit}  (threshold: {threshold}{unit})")

print("\n── Metric comparisons ──")

check("Accuracy",
      baseline["accuracy_mean"],
      candidate["accuracy_mean"],
      THRESHOLDS["accuracy_drop_max"],
      direction="lower_is_worse")

check("Hallucination rate",
      baseline["hallucination_mean"],
      candidate["hallucination_mean"],
      THRESHOLDS["hallucination_rise_max"],
      direction="higher_is_worse")

check("Latency P95",
      baseline["latency_p95"],
      candidate["latency_p95"],
      THRESHOLDS["latency_p95_rise_max"],
      direction="higher_is_worse",
      unit="ms")

# Absolute floor checks
print(f"\n── Absolute floor checks ──")

acc = candidate["accuracy_mean"]
hal = candidate["hallucination_mean"]

acc_floor = acc < THRESHOLDS["min_accuracy"]
hal_ceil  = hal > THRESHOLDS["max_hallucination"]

checks.append(acc_floor)
checks.append(hal_ceil)

print(f"\n  {'❌ FAIL' if acc_floor else '✅ PASS'} Accuracy floor")
print(f"         Candidate accuracy {acc:.1%} vs minimum {THRESHOLDS['min_accuracy']:.0%}")

print(f"\n  {'❌ FAIL' if hal_ceil else '✅ PASS'} Hallucination ceiling")
print(f"         Candidate hallucination {hal:.1%} vs maximum {THRESHOLDS['max_hallucination']:.0%}")

# ─────────────────────────────────────────────
# Final decision
# ─────────────────────────────────────────────
any_failed = any(checks)
print("\n" + "="*55)
if any_failed:
    print("  🚫 DEPLOY BLOCKED — one or more checks failed")
    print("     Fix regressions before shipping to production")
    sys.exit(1)   # non-zero exit = CI pipeline fails
else:
    print("  ✅ DEPLOY APPROVED — all checks passed")
    print("     Safe to promote candidate to production")
    sys.exit(0)
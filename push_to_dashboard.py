import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
from datetime import datetime
from evidently.ui.workspace import Workspace
from evidently.report import Report
from evidently.metrics import (
    ColumnSummaryMetric,
    ColumnDistributionMetric,
)

# ─────────────────────────────────────────────
# BLOCK 1: Load both experiment results
# ─────────────────────────────────────────────
df_v1 = pd.read_csv("experiment_v1_0.csv")
df_v2 = pd.read_csv("experiment_v2_0.csv")

with open("experiment_v1_0_config.json") as f: cfg_v1 = json.load(f)
with open("experiment_v2_0_config.json") as f: cfg_v2 = json.load(f)

print(f"Loaded v1.0: {len(df_v1)} rows")
print(f"Loaded v2.0: {len(df_v2)} rows")

# ─────────────────────────────────────────────
# BLOCK 2: Connect to local Evidently workspace
# ─────────────────────────────────────────────
ws = Workspace("./evidently_workspace")

# Find or create project
project = None
for p in ws.list_projects():
    if p.name == "llm-eval-dash":
        project = p
        break

if project is None:
    project = ws.create_project("llm-eval-dash")
    project.description = "LLM Evaluation Dashboard — all 4 metrics"
    project.save()
    print(f"✓ Created project: {project.id}")
else:
    print(f"✓ Found project:   {project.id}")

# ─────────────────────────────────────────────
# BLOCK 3: Build report for v1.0 (baseline)
# ─────────────────────────────────────────────
report_v1 = Report(
    metrics=[
        ColumnSummaryMetric(column_name="accuracy_score"),
        ColumnSummaryMetric(column_name="hallucination_rate"),
        ColumnSummaryMetric(column_name="latency_ms"),
        ColumnSummaryMetric(column_name="answer_sentiment"),
        ColumnDistributionMetric(column_name="accuracy_score"),
        ColumnDistributionMetric(column_name="hallucination_rate"),
        ColumnDistributionMetric(column_name="latency_ms"),
    ],
    timestamp=datetime(2026, 1, 1),   # fixed date so v1 appears before v2
)
report_v1.run(reference_data=None, current_data=df_v1)
report_v1.save_html("report_v1.html")
ws.add_report(project.id, report_v1)
print("✓ Pushed v1.0 report to dashboard")

# ─────────────────────────────────────────────
# BLOCK 4: Build report for v2.0 (bad prompt)
# ─────────────────────────────────────────────
report_v2 = Report(
    metrics=[
        ColumnSummaryMetric(column_name="accuracy_score"),
        ColumnSummaryMetric(column_name="hallucination_rate"),
        ColumnSummaryMetric(column_name="latency_ms"),
        ColumnSummaryMetric(column_name="answer_sentiment"),
        ColumnDistributionMetric(column_name="accuracy_score"),
        ColumnDistributionMetric(column_name="hallucination_rate"),
        ColumnDistributionMetric(column_name="latency_ms"),
    ],
    timestamp=datetime(2026, 1, 2),   # one day later so it shows as newer
)
report_v2.run(reference_data=df_v1, current_data=df_v2)  # v1 as reference = drift visible
report_v2.save_html("report_v2.html")
ws.add_report(project.id, report_v2)
print("✓ Pushed v2.0 report to dashboard")

# ─────────────────────────────────────────────
# BLOCK 5: Print project URL
# ─────────────────────────────────────────────
print(f"\n✓ Done. Open your dashboard:")
print(f"  http://127.0.0.1:8000/projects/{project.id}/reports")
print(f"\nYou should see two reports — v1.0 and v2.0")
print(f"Click each to see accuracy, hallucination, latency panels")
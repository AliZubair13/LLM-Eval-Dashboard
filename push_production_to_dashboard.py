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

# Load production traces
traces = []
with open("production_traces.jsonl") as f:
    for line in f:
        traces.append(json.loads(line))
df = pd.DataFrame(traces)
print(f"Loaded {len(df)} production traces")

# Load eval results if they exist
try:
    evals = []
    with open("production_evals.jsonl") as f:
        for line in f:
            evals.append(json.loads(line))
    evals_df = pd.DataFrame(evals)
    print(f"Loaded {len(evals_df)} eval results")
    has_evals = True
except FileNotFoundError:
    has_evals = False
    print("No eval results yet")

# Connect to workspace
ws = Workspace("./evidently_workspace")
project = None
for p in ws.list_projects():
    if p.name == "llm-eval-dash":
        project = p
        break

if project is None:
    project = ws.create_project("llm-eval-dash")
    project.save()

# Build production monitoring report
metrics = [
    ColumnSummaryMetric(column_name="latency_ms"),
    ColumnDistributionMetric(column_name="latency_ms"),
    ColumnSummaryMetric(column_name="prompt_tokens"),
    ColumnSummaryMetric(column_name="completion_tokens"),
]

report = Report(
    metrics=metrics,
    timestamp=datetime(2026, 1, 3),
)
report.run(reference_data=None, current_data=df)
report.save_html("report_production.html")
ws.add_report(project.id, report)
print("✓ Pushed production traces report")

# Push eval report if we have eval data
if has_evals:
    eval_metrics = [
        ColumnSummaryMetric(column_name="judge_score"),
        ColumnDistributionMetric(column_name="judge_score"),
        ColumnSummaryMetric(column_name="latency_ms"),
    ]
    eval_report = Report(
        metrics=eval_metrics,
        timestamp=datetime(2026, 1, 4),
    )
    eval_report.run(reference_data=None, current_data=evals_df)
    eval_report.save_html("report_production_evals.html")
    ws.add_report(project.id, eval_report)
    print("✓ Pushed production eval report")

print(f"\n✓ Open dashboard:")
print(f"  http://127.0.0.1:8000/projects/{project.id}/reports")
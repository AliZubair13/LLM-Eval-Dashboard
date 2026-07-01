# LLM Eval Dashboard

A production-grade LLM evaluation dashboard that benchmarks and monitors multiple AI models across accuracy, hallucination, latency, and user satisfaction metrics.

Built with **Evidently AI**, **OpenAI**, **Anthropic**, **Groq**, and **Streamlit**.

---

## Live Demo

> Run locally with `streamlit run dashboard.py` after setup.

---

## Dashboard Pages

| Page | Description |
|---|---|
| Model Comparison | Radar chart + scorecard across all 4 models |
| Accuracy | Category breakdown, score distribution, worst answers |
| Hallucination | Faithfulness gauges, category bars |
| Latency | Box plots, P50/P95/P99 percentile table |
| Cost Analysis | Cost vs accuracy scatter, projection table |
| Satisfaction | Sentiment proxy by category, answer length scatter |
| Production Traces | Live request monitoring, token usage, online evals |
| Raw Data | Full scored dataset per model with download |

---

## Models Evaluated

| Model | Provider | Accuracy | Hallucination | Latency P95 | Cost/query |
|---|---|---|---|---|---|
| Llama 3.1 8B ⭐ | Groq | 87.0% | 2.5% | 892ms | $0.000008 |
| GPT-4o | OpenAI | 89.0% | 1.5% | 2961ms | $0.000606 |
| GPT-4o Mini | OpenAI | 90.0% | 2.0% | 5204ms | $0.000032 |
| Claude Sonnet | Anthropic | 86.5% | 2.0% | 7407ms | $0.001455 |

> ⭐ **Llama 3.1 8B via Groq** wins on composite score (accuracy 40% + faithfulness 30% + speed 15% + cost 15%) — nearly matching GPT-4o accuracy at 75x lower cost and 3x faster latency.

---

## Screenshots

### Model Comparison
<!-- Add screenshot here -->
<img width="1284" height="537" alt="image" src="https://github.com/user-attachments/assets/dcfde15c-5cd1-4e1e-9a4d-16edcc155d36" />


### Accuracy by Category
<!-- Add screenshot here -->
![Accuracy](screenshots/accuracy.png)

### Hallucination Rates
<!-- Add screenshot here -->
![Hallucination](screenshots/hallucination.png)

### Latency Distribution
<!-- Add screenshot here -->
![Latency](screenshots/latency.png)

### Cost Analysis
<!-- Add screenshot here -->
![Cost Analysis](screenshots/cost_analysis.png)

---

## Architecture

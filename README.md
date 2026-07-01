# LLM Eval Dashboard

A production-grade LLM evaluation dashboard that benchmarks and monitors multiple AI models across accuracy, hallucination, latency, and user satisfaction metrics.

Built with **Evidently AI**, **OpenAI**, **Anthropic**, **Groq**, and **Streamlit**.

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

<img width="1302" height="515" alt="image" src="https://github.com/user-attachments/assets/af5ce913-e36f-4492-8f1f-173a22043fc9" />

<img width="808" height="581" alt="image" src="https://github.com/user-attachments/assets/9e3cf180-a5b3-467b-bc0b-7d20b19acec2" />




### Accuracy by Category
<img width="850" height="585" alt="image" src="https://github.com/user-attachments/assets/e99782fb-69f0-4a14-8d94-d0a4d729f50d" />

<img width="806" height="572" alt="image" src="https://github.com/user-attachments/assets/8901ec96-5671-4dd4-a04f-5cb2d00c3239" />



### Hallucination Rates
<img width="835" height="580" alt="image" src="https://github.com/user-attachments/assets/511add68-f917-413f-a301-b21031828bc9" />

<img width="816" height="251" alt="image" src="https://github.com/user-attachments/assets/e10ce689-8420-49e9-8f1b-e0588aeab1e6" />



### Latency Distribution
<img width="856" height="592" alt="image" src="https://github.com/user-attachments/assets/1f372fdf-a920-4b9d-b05f-15525a04587d" />

<img width="819" height="572" alt="image" src="https://github.com/user-attachments/assets/4a749fcf-539c-4083-a50d-094f2624fecd" />



### Cost Analysis
<img width="887" height="333" alt="image" src="https://github.com/user-attachments/assets/3be3ec37-5a4b-459d-9e08-4aabb72a39a0" />

<img width="873" height="440" alt="image" src="https://github.com/user-attachments/assets/786b13da-5ca3-47fb-8966-50da5e9ac253" />

<img width="848" height="540" alt="image" src="https://github.com/user-attachments/assets/34878fd2-2ca2-4c1b-8ab8-ff8567ac2d8a" />

### Satisfaction
<img width="888" height="578" alt="image" src="https://github.com/user-attachments/assets/180aa893-0066-4fef-ae82-65644373c11a" />

<img width="898" height="480" alt="image" src="https://github.com/user-attachments/assets/c2f8cdce-f5dd-4d65-af36-b0932ea7d9c3" />

### Production Traces 
<img width="851" height="506" alt="image" src="https://github.com/user-attachments/assets/f6596a72-fc15-4716-8f54-28ac7979d619" />

<img width="829" height="476" alt="image" src="https://github.com/user-attachments/assets/091bde4a-9f97-42a5-a77f-10409b1a95c5" />

<img width="857" height="370" alt="image" src="https://github.com/user-attachments/assets/dbd99cac-9215-4100-8f7e-ad9611c00f68" />


### Raw Data

<img width="854" height="614" alt="image" src="https://github.com/user-attachments/assets/d5f2fd07-eba0-4855-8011-9e516308991c" />


---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AliZubair13/LLM-Eval-Dashboard
cd LLM-Eval-Dashboard
```

### 2. Create virtual environment

```bash
py -m venv venv
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate          # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set API keys

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-proj-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:GROQ_API_KEY="gsk_..."

# Mac/Linux
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
```

### 5. Run all 4 model experiments

```bash
python run_experiment.py
```

### 6. Launch the dashboard

```bash
python -m streamlit run dashboard.py
```

Opens at `http://localhost:8501`

---

## Cost to Run Full Benchmark

| Model | 20 questions cost |
|---|---|
| Llama 3.1 8B | $0.0002 (Groq free tier) |
| GPT-4o Mini | $0.0006 |
| GPT-4o | $0.0121 |
| Claude Sonnet | $0.0291 |
| **Total** | **~$0.042** |

---

## Key Findings

- **Speed:** Llama 3.1 8B is 3.3x faster than GPT-4o and 8.3x faster than Claude Sonnet at P95
- **Cost:** Llama 3.1 8B costs 75x less than GPT-4o and 181x less than Claude Sonnet per query
- **Accuracy gap:** Only 3.5% separates the best (GPT-4o Mini 90%) from the worst (Claude Sonnet 86.5%) — a very small margin for such a large cost difference
- **Hallucination:** All models perform well; GPT-4o has the lowest rate at 1.5%
- **Best value:** Llama 3.1 8B via Groq — production-ready quality at near-zero cost

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM providers | OpenAI, Anthropic, Groq |
| Tracing | Evidently AI (tracely) |
| Evaluation | LLM-as-judge (gpt-4o-mini) |
| Dashboard | Streamlit + Plotly |
| Data | Pandas, CSV, JSONL |
| CI gate | Python script (regression_gate.py) |

---





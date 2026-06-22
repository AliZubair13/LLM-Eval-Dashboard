import pandas as pd

# These are the actual answers your model gave in Step 1
# In production you'd pull these from your DB or Evidently trace export
production_traces = [
    {
        "question": "What is the boiling point of water?",
        "context": "",
        "ground_truth": "100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
        "category": "factual",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "context": "",
        "ground_truth": "Leonardo da Vinci.",
        "category": "factual",
    },
    {
        "question": "What is the capital of France?",
        "context": "",
        "ground_truth": "Paris.",
        "category": "factual",
    },
]

# Load existing dataset and append
existing = pd.read_csv("eval_dataset_v1.csv")
new_rows = pd.DataFrame(production_traces)

# Deduplicate by question so no row appears twice
combined = pd.concat([existing, new_rows]).drop_duplicates(subset=["question"])
combined.to_csv("eval_dataset_v1.csv", index=False)

print(f"✓ Dataset updated: {len(combined)} total examples")
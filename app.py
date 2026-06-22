import os, time
from tracely import init_tracing, trace_event, get_current_span
from openai import OpenAI

init_tracing(
    address="http://localhost:8000",          # local UI; or "https://app.evidently.cloud"
    api_key=os.environ.get("EVIDENTLY_API_KEY", ""),  # leave blank for local
    project_id="019ebe26-12f3-7feb-af46-f98c2b6e9dfd",
    export_name="llm-prod-traces",
)

client = OpenAI()   # needs OPENAI_API_KEY set

@trace_event()
def answer_question(question: str) -> str:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    span = get_current_span()
    span.set_attribute("latency_ms", round(latency_ms, 1))
    span.set_attribute("model", "gpt-4o-mini")
    span.set_attribute("prompt_version", "v2.3")
    return resp.choices[0].message.content

if __name__ == "__main__":
    # this is what actually generates traces
    print(answer_question("What is the boiling point of water?"))
    print(answer_question("Who painted the Mona Lisa?"))
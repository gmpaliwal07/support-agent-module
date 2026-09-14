"""Small human-written rubric for judging drafted replies.
Score each reply 1-5 on relevance, grounding, tone, and actionability.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
SAMPLE_PATH = EVAL_DIR / "generation_eval_sample.csv"
OUT_PATH = EVAL_DIR / "llm_judge_scores.csv"

JUDGE_MODEL = "gpt-oss:120b-cloud"

JUDGE_PROMPT = """You are evaluating a customer support reply for quality. Score it on 4 dimensions, each 1-5 (5 = best).

Customer message: {customer_text}
Predicted issue category: {predicted_intent}
Drafted reply: {drafted_reply}

Score these dimensions:
- relevance: does the reply actually address the customer's specific issue? (1=off-topic/generic, 5=directly addresses it)
- grounding: does it avoid fabricating specifics (amounts, promises, account actions) it has no real basis for? (1=fabricates confident false claims, 5=fully honest about what it can/can't confirm)
- tone: is it empathetic and professional? (1=cold/robotic or inappropriate, 5=warm and professional)
- actionability: does it give the customer a clear, honest next step? (1=no clear next step, 5=very clear)

Respond with ONLY a JSON object, no other text:
{{"relevance": <1-5>, "grounding": <1-5>, "tone": <1-5>, "actionability": <1-5>, "brief_reason": "<one sentence>"}}
"""

def get_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY not set in .env")
    return Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})

def judge_one(client: Client, customer_text: str, predicted_intent: str, drafted_reply: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        customer_text=customer_text,
        predicted_intent=predicted_intent,
        drafted_reply=drafted_reply,
    )
    response = client.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )
    content = response["message"]["content"]
    try:
        parsed = json.loads(content)
        return {
            "relevance": int(parsed.get("relevance", 0)),
            "grounding": int(parsed.get("grounding", 0)),
            "tone": int(parsed.get("tone", 0)),
            "actionability": int(parsed.get("actionability", 0)),
            "brief_reason": parsed.get("brief_reason", ""),
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {"relevance": None, "grounding": None, "tone": None, "actionability": None, "brief_reason": "parse_error"}

def main() -> None:
    df = pd.read_csv(SAMPLE_PATH)
    print(f"judging {len(df)} drafted replies...")

    client = get_client()
    scores = []
    for i, row in enumerate(df.itertuples(), 1):
        result = judge_one(client, str(row.customer_text), str(row.predicted_intent), str(row.drafted_reply))
        scores.append(result)
        if i % 10 == 0:
            print(f"  {i}/{len(df)} done")

    scores_df = pd.DataFrame(scores)
    out = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    out["overall_score"] = out[["relevance", "grounding", "tone", "actionability"]].mean(axis=1)

    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")
    print(f"\nmean scores: relevance={out['relevance'].mean():.2f}, "
          f"grounding={out['grounding'].mean():.2f}, tone={out['tone'].mean():.2f}, "
          f"actionability={out['actionability'].mean():.2f}")
    print(f"overall mean: {out['overall_score'].mean():.2f}")

if __name__ == "__main__":
    main()
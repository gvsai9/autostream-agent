"""
Intent Classifier — The Cognitive Router
------------------------------------------
Uses Gemini 1.5 Flash via langchain_google_genai to classify every user
message into a structured JSON with 4 fields:
    - intent:    GREETING | PRODUCT_INQUIRY | HIGH_INTENT | OUT_OF_DOMAIN
    - sentiment: excited | neutral | frustrated | curious
    - entities:  any extracted info (platform, plan, name, email)
    - reasoning: why this intent was chosen (useful for logging)
"""

import json
import logging
import re
import os
import sys
import time
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Allow importing config from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, get_llm_with_fallback

logger = logging.getLogger(__name__)


# ── Data contract for what the classifier returns ────────────────────────────

class ClassificationResult(TypedDict):
    intent: str
    sentiment: str
    entities: dict
    reasoning: str


# ── The classification prompt — NOTE: uses __MESSAGE__ and __HISTORY__
#    to avoid conflicts with Python's .format() and curly braces in JSON ──────

CLASSIFIER_PROMPT_TEMPLATE = """
You are an intent classification engine for AutoStream, a video editing SaaS.
Your ONLY job is to analyze the user message and return a JSON object.

## Intent Labels (pick exactly one):
- GREETING        : Simple hello, hi, hey, how are you, etc.
- PRODUCT_INQUIRY : Asking about features, pricing, plans, refunds, support, trials
- HIGH_INTENT     : User clearly wants to sign up, try, purchase, or subscribe
- OUT_OF_DOMAIN   : Anything unrelated to AutoStream (poems, coding help, competitor comparisons, etc.)

## Sentiment Labels (pick exactly one):
- excited, neutral, frustrated, curious

## Entity Extraction:
Extract any of these if present in the message:
- platform: YouTube, Instagram, TikTok, Facebook, LinkedIn, or other
- plan_interest: Basic or Pro
- name: if user mentions their name
- email: if user mentions their email address

## Rules:
1. Return ONLY valid JSON. No markdown, no backticks, no text outside the JSON.
2. If no entities found, use an empty JSON object for entities field.
3. HIGH_INTENT requires clear purchase/signup signals. "Sounds good" alone is NOT high intent.

## Output Format (return exactly this structure):
{
  "intent": "INTENT_LABEL",
  "sentiment": "sentiment_label",
  "entities": {},
  "reasoning": "one sentence explaining why"
}

## User Message:
"__MESSAGE__"

## Conversation History (last 3 turns for context):
__HISTORY__

Return the JSON object now:
"""


# ── Classifier function ───────────────────────────────────────────────────────

def classify_message(
    message: str,
    conversation_history: list,
    api_key: str
) -> ClassificationResult:
    """
    Sends the user message to Gemini and returns structured classification.

    Args:
        message: Latest user message
        conversation_history: List of previous {"role": ..., "content": ...} dicts
        api_key: Gemini API key

    Returns:
        ClassificationResult dict with intent, sentiment, entities, reasoning
    """
    llm = get_llm_with_fallback(api_key, temperature=0)

    # Brief pause to avoid hammering the per-minute rate limiter
    time.sleep(1)

    # Format last 3 turns for context (keeps prompt short)
    recent = conversation_history[-3:] if conversation_history else []
    history_str = "\n".join([
        f"{t['role'].upper()}: {t['content']}" for t in recent
    ]) if recent else "No prior conversation."

    # Use simple string replacement — avoids .format() conflicts with JSON braces
    prompt = CLASSIFIER_PROMPT_TEMPLATE.replace("__MESSAGE__", message).replace("__HISTORY__", history_str)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if model wraps in ```json ... ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result: ClassificationResult = json.loads(raw)

        # Validate required keys
        for key in ["intent", "sentiment", "entities", "reasoning"]:
            if key not in result:
                raise ValueError(f"Missing key: {key}")

        # Telemetry log — visible in terminal during demo
        logger.info(
            f"[INTENT] {result['intent']} | "
            f"Sentiment: {result['sentiment']} | "
            f"Entities: {result['entities']} | "
            f"Reason: {result['reasoning']}"
        )

        return result

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[INTENT] Classification failed: {e}. Defaulting to PRODUCT_INQUIRY.")
        return ClassificationResult(
            intent="PRODUCT_INQUIRY",
            sentiment="neutral",
            entities={},
            reasoning="Classification failed, defaulted safely"
        )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv("GEMINI_API_KEY")

    tests = [
        ("Hi there!", []),
        ("What is the price of the Pro plan?", []),
        ("I want to sign up for Pro for my YouTube channel.", []),
        ("Write me a poem about 4K resolution.", []),
    ]

    for msg, hist in tests:
        print(f"\nMessage: {msg}")
        r = classify_message(msg, hist, api_key)
        print(f"  Intent:    {r['intent']}")
        print(f"  Sentiment: {r['sentiment']}")
        print(f"  Entities:  {r['entities']}")
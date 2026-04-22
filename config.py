"""
Config — Central settings for the AutoStream agent.
Load API keys from environment variables (never hardcode keys in source).
"""

import logging
import os
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # loads from .env file if present

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "\n\n❌ GEMINI_API_KEY not found!\n"
        "Please set it:\n"
        "  Option 1: Create a .env file with: GEMINI_API_KEY=your_key_here\n"
        "  Option 2: export GEMINI_API_KEY=your_key_here\n"
        "Get your free key at: https://aistudio.google.com/app/apikey\n"
    )

# Model settings — ordered by preference (fastest/cheapest first)
# get_llm_with_fallback() tries them in this order when rate-limited
LLM_MODEL = "gemini-2.5-flash-lite"   # primary   — fastest, lowest quota cost
LLM_FALLBACK_MODELS = [
    "gemini-2.5-flash",                              # fallback 1 — still fast, higher quota
    "gemini-2.0",                              # fallback 2 — most capable, use last
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG settings
RAG_TOP_K = 2  # number of chunks to retrieve per query


# ── Fallback LLM Factory ──────────────────────────────────────────────────────

def get_llm_with_fallback(api_key: str, temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """
    Returns a working Gemini LLM instance, trying models in fallback order.

    Fallback chain:
        gemini-2.5-flash-lite  →  gemini-2.0-flash  →  gemini-2.5-flash

    On a 429 / resource-exhausted error the function:
      1. Logs a warning so the terminal shows which model was swapped
      2. Waits 2 seconds (gives the quota window a moment to breathe)
      3. Retries with the next model in the chain

    All other errors (bad key, network, etc.) are re-raised immediately
    because a fallback model won't fix them.

    Args:
        api_key:     Gemini API key from environment
        temperature: Sampling temperature (0 = deterministic, 0.7 = creative)

    Returns:
        The first ChatGoogleGenerativeAI instance that responds without a
        rate-limit error.

    Raises:
        RuntimeError: If every model in the chain is exhausted.
    """
    all_models = [LLM_MODEL] + LLM_FALLBACK_MODELS

    for index, model in enumerate(all_models):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                max_retries=0,          # we handle retries ourselves across models
            )

            # Cheap probe — invoke with an empty message list to validate the
            # model is reachable before handing it to a real node.
            # We pass a minimal ping rather than a real prompt so it costs
            # essentially zero tokens.
            llm.invoke([{"role": "user", "content": "ping"}])

            if index > 0:
                logger.warning(
                    f"[LLM FALLBACK] '{all_models[index - 1]}' rate-limited. "
                    f"Now using '{model}'."
                )
            else:
                logger.info(f"[LLM] Using primary model: '{model}'")

            return llm

        except Exception as exc:
            error_str = str(exc).lower()
            is_rate_limit = any(
                token in error_str
                for token in ("429", "resource_exhausted", "quota", "rate","503","404")
            )

            if is_rate_limit:
                logger.warning(
                    f"[LLM FALLBACK] '{model}' is rate-limited "
                    f"({type(exc).__name__}). "
                    f"Trying next model in 2 s..."
                )
                time.sleep(2)
                continue  # try the next model

            # Non-rate-limit error (bad key, network, etc.) — no point falling back
            logger.error(f"[LLM] Non-recoverable error on '{model}': {exc}")
            raise

    raise RuntimeError(
        "[LLM FALLBACK] All Gemini models exhausted:\n"
        + "\n".join(f"  • {m}" for m in all_models)
        + "\nWait ~60 s for quota to reset, then try again."
    )
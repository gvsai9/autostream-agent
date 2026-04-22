"""
LangGraph State Machine — The Core Agent
-----------------------------------------
Graph Flow:
    START
      |
  [analyze_input]   <- classify intent + extract entities
      |
  [route_intent]    <- conditional edge picks next node
      |
  GREETING -> [greet_node]
  PRODUCT_INQUIRY -> [rag_node]
  HIGH_INTENT -> [collect_lead_node] -> [ask_field_node] (loop) -> [capture_lead_node]
  OUT_OF_DOMAIN -> [out_of_domain_node]
"""

import logging
import os
import sys
import time
from typing import TypedDict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_MODEL, get_llm_with_fallback
from agent.intent import classify_message
from agent.rag import retrieve_context
from agent.tools import mock_lead_capture, get_missing_lead_fields, extract_email_from_text

logger = logging.getLogger(__name__)


# ── Prompts (inline to keep graph self-contained) ────────────────────────────

GREET_PROMPT = """You are AutoStream's friendly AI assistant.
The user just greeted you. Respond warmly in 2-3 sentences.
Let them know you can help with pricing, features, and getting started.
Do not list all features yet — keep it welcoming."""

RAG_PROMPT_TEMPLATE = """You are AutoStream's knowledgeable product assistant.
Answer the user's question using ONLY the context below.
Do not make up features, prices, or policies not in the context.
Be concise, friendly, and clear. Use bullet points for features/pricing.
At the end, subtly invite further interest.

Context from Knowledge Base:
__CONTEXT__

User Question: __QUESTION__

Conversation History:
__HISTORY__"""

LEAD_ASK_TEMPLATE = """You are AutoStream's onboarding assistant.
The user wants to sign up. Collect their details one at a time.

Already collected:
- Name: __NAME__
- Email: __EMAIL__
- Platform: __PLATFORM__

Next field needed: __NEXT__

Ask ONLY for the next missing field in one friendly sentence.
Do NOT ask for multiple things at once."""

LEAD_SUCCESS_TEMPLATE = """You are AutoStream's onboarding assistant.
The user has just been registered. Their details:
- Name: __NAME__
- Email: __EMAIL__
- Platform: __PLATFORM__

Write a warm 2-3 sentence confirmation. Tell them they'll receive
a trial link at their email. Make it enthusiastic!"""

OOD_PROMPT = """You are AutoStream's AI assistant.
The user asked something outside your scope.
Politely decline in 1-2 sentences and smoothly redirect back to
AutoStream's video editing features and plans."""

FRUSTRATED_NOTE = "\nNote: The user seems frustrated. Lead with empathy before answering."


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list
    current_input: str
    intent: str
    sentiment: str
    entities: dict
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool   # True from first HIGH_INTENT until capture — guards mid-flow misclassification
    rag_context: str
    response: str


def get_llm(api_key: str, temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Delegates to the centralized fallback factory in config.py."""
    return get_llm_with_fallback(api_key, temperature=temperature)


# ── Node functions ────────────────────────────────────────────────────────────

def analyze_input_node(state: AgentState) -> AgentState:
    """Classify intent, extract sentiment + entities from user message."""
    api_key = os.getenv("GEMINI_API_KEY")
    result = classify_message(state["current_input"], state["messages"], api_key)

    entities = result.get("entities", {})

    # Entity pre-fill: if user mentioned platform/name/email, save immediately
    updated_name = state.get("lead_name") or entities.get("name")
    updated_email = state.get("lead_email") or entities.get("email")
    updated_platform = state.get("lead_platform") or entities.get("platform")

    logger.info(
        f"\n[STATE TRANSITION] Intent -> {result['intent']} | "
        f"Sentiment -> {result['sentiment']} | "
        f"Entities -> {entities}"
    )

    return {
        **state,
        "intent": result["intent"],
        "sentiment": result["sentiment"],
        "entities": entities,
        "lead_name": updated_name,
        "lead_email": updated_email,
        "lead_platform": updated_platform,
        # Latch to True on first HIGH_INTENT — never goes back to False until capture
        "collecting_lead": state.get("collecting_lead") or result["intent"] == "HIGH_INTENT",
    }


def greet_node(state: AgentState) -> AgentState:
    """Warm greeting response."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = get_llm(api_key)
    prompt = GREET_PROMPT
    if state.get("sentiment") == "frustrated":
        prompt += FRUSTRATED_NOTE
    time.sleep(1)
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.info("[NODE] greet_node")
    return {**state, "response": response.content.strip()}


def rag_node(state: AgentState) -> AgentState:
    """Retrieve from KB and answer product questions."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = get_llm(api_key)

    context = retrieve_context(state["current_input"])

    history_str = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in state["messages"][-4:]
    ]) or "No prior conversation."

    prompt = (RAG_PROMPT_TEMPLATE
              .replace("__CONTEXT__", context)
              .replace("__QUESTION__", state["current_input"])
              .replace("__HISTORY__", history_str))

    if state.get("sentiment") == "frustrated":
        prompt += FRUSTRATED_NOTE

    time.sleep(1)
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.info("[NODE] rag_node")
    return {**state, "rag_context": context, "response": response.content.strip()}


def collect_lead_node(state: AgentState) -> AgentState:
    """Extract lead fields from the user's latest message."""
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")
    user_input = state["current_input"].strip()

    missing = get_missing_lead_fields(name, email, platform)

    if missing:
        asking_for = missing[0]

        if asking_for == "name" and not name:
            # Accept as name only if short, no email symbol, not a sentence
            words = user_input.split()
            if 1 <= len(words) <= 4 and "@" not in user_input and len(user_input) < 40:
                name = user_input.title()

        elif asking_for == "email" and not email:
            extracted = extract_email_from_text(user_input)
            if extracted:
                email = extracted
            elif "@" in user_input:
                email = user_input.strip()

        elif asking_for == "platform" and not platform:
            # Only accept known platforms or very short answers (1-2 words)
            known = ["youtube", "instagram", "tiktok", "facebook", "linkedin", "twitter", "x", "twitch"]
            for p in known:
                if p in user_input.lower():
                    platform = p.capitalize()
                    break
            # Accept short free-form answers like "Twitch" but NOT sentences
            if not platform and len(user_input.split()) <= 2:
                platform = user_input.strip().capitalize()
            # else: platform stays None — agent will ask again properly

    logger.info(
        f"[NODE] collect_lead | Name:{name} Email:{email} Platform:{platform} | "
        f"Still missing: {get_missing_lead_fields(name, email, platform)}"
    )

    return {**state, "lead_name": name, "lead_email": email, "lead_platform": platform, "response": ""}


def ask_field_node(state: AgentState) -> AgentState:
    """Ask for the next missing lead field naturally."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = get_llm(api_key)

    missing = get_missing_lead_fields(
        state.get("lead_name"),
        state.get("lead_email"),
        state.get("lead_platform")
    )
    next_field = missing[0] if missing else "none"

    prompt = (LEAD_ASK_TEMPLATE
              .replace("__NAME__", state.get("lead_name") or "Not yet provided")
              .replace("__EMAIL__", state.get("lead_email") or "Not yet provided")
              .replace("__PLATFORM__", state.get("lead_platform") or "Not yet provided")
              .replace("__NEXT__", next_field))

    time.sleep(1)
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.info(f"[NODE] ask_field_node — asking for: {next_field}")
    return {**state, "response": response.content.strip()}


def capture_lead_node(state: AgentState) -> AgentState:
    """All fields collected — fire the tool and send confirmation."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = get_llm(api_key)

    # Fire the mock lead capture tool
    mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"]
    )

    prompt = (LEAD_SUCCESS_TEMPLATE
              .replace("__NAME__", state["lead_name"])
              .replace("__EMAIL__", state["lead_email"])
              .replace("__PLATFORM__", state["lead_platform"]))

    time.sleep(1)
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.info("[NODE] capture_lead_node — lead captured!")
    return {**state, "lead_captured": True, "response": response.content.strip()}


def out_of_domain_node(state: AgentState) -> AgentState:
    """Politely decline off-topic questions."""
    api_key = os.getenv("GEMINI_API_KEY")
    llm = get_llm(api_key)
    time.sleep(1)
    response = llm.invoke([HumanMessage(content=OOD_PROMPT)])
    logger.info("[NODE] out_of_domain_node")
    return {**state, "response": response.content.strip()}


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "PRODUCT_INQUIRY")

    # If lead collection is in progress — either the flag is set (catches the
    # case where name/email/platform are still None, e.g. first reply after
    # HIGH_INTENT) OR any field is already populated — keep routing to
    # collect_lead_node regardless of what the classifier said this turn.
    lead_in_progress = (
        state.get("collecting_lead")
        or any([state.get("lead_name"), state.get("lead_email"), state.get("lead_platform")])
    )
    if lead_in_progress and not state.get("lead_captured"):
        return "collect_lead_node"

    routes = {
        "GREETING": "greet_node",
        "PRODUCT_INQUIRY": "rag_node",
        "HIGH_INTENT": "collect_lead_node",
        "OUT_OF_DOMAIN": "out_of_domain_node",
    }
    return routes.get(intent, "rag_node")


def route_after_collect(state: AgentState) -> str:
    missing = get_missing_lead_fields(
        state.get("lead_name"),
        state.get("lead_email"),
        state.get("lead_platform")
    )
    return "ask_field_node" if missing else "capture_lead_node"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("greet_node", greet_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("collect_lead_node", collect_lead_node)
    graph.add_node("ask_field_node", ask_field_node)
    graph.add_node("capture_lead_node", capture_lead_node)
    graph.add_node("out_of_domain_node", out_of_domain_node)

    graph.set_entry_point("analyze_input")

    graph.add_conditional_edges(
        "analyze_input",
        route_by_intent,
        {
            "greet_node": "greet_node",
            "rag_node": "rag_node",
            "collect_lead_node": "collect_lead_node",
            "out_of_domain_node": "out_of_domain_node",
        }
    )

    graph.add_conditional_edges(
        "collect_lead_node",
        route_after_collect,
        {
            "ask_field_node": "ask_field_node",
            "capture_lead_node": "capture_lead_node",
        }
    )

    graph.add_edge("greet_node", END)
    graph.add_edge("rag_node", END)
    graph.add_edge("ask_field_node", END)
    graph.add_edge("capture_lead_node", END)
    graph.add_edge("out_of_domain_node", END)

    return graph.compile()
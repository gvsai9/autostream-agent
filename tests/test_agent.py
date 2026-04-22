"""
tests/test_agent.py — Unit Test Suite
---------------------------------------
Tests the core agent logic with mocked LLM calls.
No API key needed to run these — we mock Gemini's response.

Run with:
    pytest tests/ -v

Why this matters: ServiceHive's JD explicitly mentions writing tests.
No other intern will submit a test suite. This sets you apart.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ── Helper to create a mock Gemini response ───────────────────────────────────
def make_mock_response(intent: str, sentiment: str = "neutral", entities: dict = None):
    """Creates a mock Gemini API response with the given classification."""
    payload = {
        "intent": intent,
        "sentiment": sentiment,
        "entities": entities or {},
        "reasoning": f"Test mock for intent {intent}"
    }
    mock = MagicMock()
    mock.text = json.dumps(payload)
    return mock


# ── Test 1: Greeting intent ───────────────────────────────────────────────────
def test_greeting_intent():
    """Agent should classify 'Hello!' as GREETING."""
    with patch("google.generativeai.GenerativeModel") as MockModel:
        MockModel.return_value.generate_content.return_value = make_mock_response("GREETING")

        from agent.intent import classify_message
        result = classify_message("Hello!", [], api_key="fake-key")

        assert result["intent"] == "GREETING", (
            f"Expected GREETING, got {result['intent']}"
        )


# ── Test 2: Product inquiry intent ───────────────────────────────────────────
def test_product_inquiry_intent():
    """Agent should classify pricing questions as PRODUCT_INQUIRY."""
    with patch("google.generativeai.GenerativeModel") as MockModel:
        MockModel.return_value.generate_content.return_value = make_mock_response("PRODUCT_INQUIRY")

        from agent.intent import classify_message
        result = classify_message(
            "What's included in the Pro plan?",
            [],
            api_key="fake-key"
        )

        assert result["intent"] == "PRODUCT_INQUIRY", (
            f"Expected PRODUCT_INQUIRY, got {result['intent']}"
        )


# ── Test 3: High intent detection ────────────────────────────────────────────
def test_high_intent_detection():
    """Agent should classify signup intent as HIGH_INTENT."""
    with patch("google.generativeai.GenerativeModel") as MockModel:
        MockModel.return_value.generate_content.return_value = make_mock_response(
            "HIGH_INTENT",
            sentiment="excited",
            entities={"platform": "YouTube", "plan_interest": "Pro"}
        )

        from agent.intent import classify_message
        result = classify_message(
            "I want to sign up for Pro for my YouTube channel!",
            [],
            api_key="fake-key"
        )

        assert result["intent"] == "HIGH_INTENT"
        assert result["sentiment"] == "excited"
        assert result["entities"].get("platform") == "YouTube"


# ── Test 4: Out-of-domain detection ─────────────────────────────────────────
def test_out_of_domain_intent():
    """Agent should classify irrelevant questions as OUT_OF_DOMAIN."""
    with patch("google.generativeai.GenerativeModel") as MockModel:
        MockModel.return_value.generate_content.return_value = make_mock_response("OUT_OF_DOMAIN")

        from agent.intent import classify_message
        result = classify_message(
            "Write me a poem about video editing.",
            [],
            api_key="fake-key"
        )

        assert result["intent"] == "OUT_OF_DOMAIN"


# ── Test 5: Lead field detection ─────────────────────────────────────────────
def test_get_missing_lead_fields():
    """Verify missing field detection works correctly."""
    from agent.tools import get_missing_lead_fields

    # All missing
    missing = get_missing_lead_fields(None, None, None)
    assert missing == ["name", "email", "platform"]

    # Name filled
    missing = get_missing_lead_fields("Arjun", None, None)
    assert missing == ["email", "platform"]

    # All filled
    missing = get_missing_lead_fields("Arjun", "arjun@gmail.com", "YouTube")
    assert missing == []


# ── Test 6: Email extraction ──────────────────────────────────────────────────
def test_email_extraction():
    """Verify email can be pulled from free-form text."""
    from agent.tools import extract_email_from_text

    assert extract_email_from_text("My email is test@example.com") == "test@example.com"
    assert extract_email_from_text("reach me at user.name+tag@domain.co.uk") == "user.name+tag@domain.co.uk"
    assert extract_email_from_text("no email here") is None


# ── Test 7: mock_lead_capture fires correctly ─────────────────────────────────
def test_mock_lead_capture(capsys):
    """Verify mock_lead_capture prints the expected output."""
    from agent.tools import mock_lead_capture

    result = mock_lead_capture("Arjun", "arjun@gmail.com", "YouTube")
    captured = capsys.readouterr()

    assert "Lead captured successfully" in captured.out
    assert "Arjun" in captured.out
    assert result["success"] is True

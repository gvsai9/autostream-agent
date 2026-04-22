"""
Tools — Agent Action Layer
---------------------------
Contains the mock_lead_capture function exactly as specified
in the assignment, plus validation guards to ensure it's
never called prematurely (missing fields = agent asks again).
"""

import logging
import re

logger = logging.getLogger(__name__)


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a qualified lead into a CRM system.
    In production, this would be a POST request to a leads API.

    Args:
        name: Lead's full name
        email: Lead's email address
        platform: Creator's primary platform (YouTube, Instagram, etc.)

    Returns:
        dict with success status and captured data
    """
    # Print exactly as required by the assignment spec
    print(f"Lead captured successfully: {name}, {email}, {platform}")

    logger.info(
        f"[TOOL] mock_lead_capture fired | "
        f"Name: {name} | Email: {email} | Platform: {platform}"
    )

    return {
        "success": True,
        "lead": {
            "name": name,
            "email": email,
            "platform": platform
        }
    }


def is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def extract_email_from_text(text: str) -> str | None:
    """Pulls an email address out of free-form user text."""
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)
    return match.group(0) if match else None


def get_missing_lead_fields(name: str | None, email: str | None, platform: str | None) -> list[str]:
    """
    Returns a list of which lead fields are still missing.
    The agent uses this to ask for the NEXT field, one at a time.
    """
    missing = []
    if not name:
        missing.append("name")
    if not email:
        missing.append("email")
    if not platform:
        missing.append("platform")
    return missing

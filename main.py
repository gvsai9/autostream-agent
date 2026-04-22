"""
main.py — AutoStream Agent CLI
"""

import os
import sys
import logging

# Suppress noisy HTTP/HuggingFace logs — keep terminal clean for demo
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

# Color codes
CYAN  = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        return f"{YELLOW}{DIM}{super().format(record)}{RESET}"


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def print_banner():
    print(f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════════════╗
║        AutoStream AI Agent — Powered by Inflx        ║
║     Social-to-Lead Agentic Workflow (ServiceHive)    ║
╚══════════════════════════════════════════════════════╝
{RESET}
{DIM}Type your message and press Enter. Type 'quit' to exit.{RESET}
    """)


def run_agent():
    setup_logging()

    # Validate API key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print(f"{RED}❌ GEMINI_API_KEY not found! Add it to your .env file.{RESET}")
        sys.exit(1)

    from agent.graph import build_agent_graph, AgentState

    print_banner()
    print(f"{DIM}[System] Compiling agent graph...{RESET}")
    agent = build_agent_graph()
    print(f"{GREEN}[System] Agent ready. Start chatting!{RESET}\n")

    # Initial state — persists across ALL turns
    state: AgentState = {
        "messages": [],
        "current_input": "",
        "intent": "",
        "sentiment": "neutral",
        "entities": {},
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
        "rag_context": "",
        "response": "",
    }

    while True:
        try:
            user_input = input(f"\n{BOLD}You: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}[System] Goodbye!{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print(f"{CYAN}Agent: Thanks for chatting! Goodbye 👋{RESET}\n")
            break

        state["current_input"] = user_input

        try:
            state = agent.invoke(state)
        except Exception as e:
            print(f"{RED}[Error] {e}{RESET}")
            logging.exception("[ERROR] Agent failed")
            continue

        print(f"\n{CYAN}{BOLD}Agent:{RESET} {CYAN}{state['response']}{RESET}")

        # Add to conversation history
        state["messages"].append({"role": "user", "content": user_input})
        state["messages"].append({"role": "assistant", "content": state["response"]})

        # Show lead capture success
        if state.get("lead_captured"):
            print(f"\n{GREEN}{BOLD}✅ Lead successfully captured!{RESET}")
            print(f"{DIM}   Name:     {state['lead_name']}")
            print(f"   Email:    {state['lead_email']}")
            print(f"   Platform: {state['lead_platform']}{RESET}\n")
            state["lead_captured"] = False   # reset so conversation continues
            state["collecting_lead"] = False  # allow fresh lead flow if user wants to sign up again


if __name__ == "__main__":
    run_agent()

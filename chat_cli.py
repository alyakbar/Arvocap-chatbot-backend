#!/usr/bin/env python3
"""
Simple terminal chat client for the Arvocap chatbot.
This runs independently from other scripts and reuses ChatbotInterface.

Usage examples:
  python chat_cli.py                # Uses OpenAI by default
  python chat_cli.py --local        # Uses local model if available
  python chat_cli.py --model .\trained_model  # Specify local model path
"""

import argparse
from chatbot_trainer import ChatbotInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arvocap Chatbot - Terminal Chat")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--openai", dest="openai", action="store_true", help="Use OpenAI model (default)")
    group.add_argument("--local", dest="openai", action="store_false", help="Use locally saved model")
    parser.set_defaults(openai=True)

    parser.add_argument(
        "--model",
        dest="model_path",
        default=None,
        help="Path to local model directory (used only with --local)"
    )
    return parser.parse_args()


def run_chat(openai: bool, model_path: str | None) -> None:
    chatbot = ChatbotInterface(use_openai=openai, model_path=model_path)

    print("\nArvocap Chatbot (terminal)\n---------------------------")
    print("Type 'quit' to exit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", ":q"}:
                print("Goodbye.")
                break

            reply = chatbot.generate_response(user_input)
            print(f"Bot: {reply}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    args = parse_args()
    run_chat(openai=args.openai, model_path=args.model_path)

#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from paper_agent.config import EmailConfig
from paper_agent.mailer.email_client import EmailClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a markdown email using EmailClient."
    )
    parser.add_argument(
        "--subject",
        default="EmailClient test message",
        help="Email subject (default: %(default)s)",
    )
    parser.add_argument(
        "--body",
        help="Inline markdown body to send. Mutually exclusive with --body-file.",
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        help="Path to a markdown file to send as the email body.",
    )
    parser.add_argument(
        "--receiver",
        help="Override receiver email. Falls back to EMAIL_RECEIVER env var if unset.",
    )

    args = parser.parse_args()

    if args.body and args.body_file:
        parser.error("Only one of --body or --body-file can be specified.")

    return args


def load_body(args: argparse.Namespace) -> str:
    if args.body_file:
        return args.body_file.read_text(encoding="utf-8")
    if args.body:
        return args.body
    return "Hello from EmailClient!\n\nThis message was sent using markdown support."


def main() -> None:
    args = parse_args()
    body = load_body(args)

    config = EmailConfig()
    print(config)
    client = EmailClient(config)
    client.send_markdown_email(args.subject, body, receiver=args.receiver)
    print("Email sent successfully.")


if __name__ == "__main__":
    main()


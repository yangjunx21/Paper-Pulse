from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

_ENV_FILE = os.getenv("PAPER_AGENT_ENV_FILE")
if _ENV_FILE:
    load_dotenv(_ENV_FILE)
else:
    load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


@dataclass
class LLMConfig:
    api_key: str = os.environ["OPENAI_API_KEY"]
    model: str = DEFAULT_MODEL
    api_base: str | None = os.getenv("OPENAI_BASE_URL")
    timeout: float = float(os.getenv("OPENAI_TIMEOUT", "30"))


@dataclass
class EmailConfig:
    host: str = os.environ["EMAIL_HOST"]
    port: int = int(os.getenv("EMAIL_PORT", "587"))
    username: str = os.environ["EMAIL_USERNAME"]
    password: str = os.environ["EMAIL_PASSWORD"]
    sender: str = os.environ["EMAIL_SENDER"]
    sender_name: str | None = os.getenv("EMAIL_SENDER_NAME")
    default_receiver: str | None = os.getenv("EMAIL_RECEIVER")
    use_tls: bool = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
    use_ssl: bool = os.getenv("EMAIL_USE_SSL", "false").lower() == "true"


def get_optional_email_config() -> EmailConfig | None:
    try:
        return EmailConfig()
    except KeyError:
        return None


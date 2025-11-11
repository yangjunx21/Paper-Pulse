from __future__ import annotations

import html
import re
import smtplib
import ssl
from typing import Optional

from email.message import EmailMessage
from email.utils import formataddr

from email_validator import EmailNotValidError, validate_email

from ..config import EmailConfig


class EmailClient:
    def __init__(self, config: EmailConfig) -> None:
        self.config = config

    def send_markdown_email(
        self,
        subject: str,
        body_markdown: str,
        receiver: Optional[str] = None,
    ) -> None:
        target = receiver or self.config.default_receiver

        validate_email(target)

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = (
            formataddr((self.config.sender_name, self.config.sender))
            if self.config.sender_name
            else self.config.sender
        )
        message["To"] = target
        message.set_content(body_markdown)

        html_body = self._markdown_to_html(body_markdown)
        message.add_alternative(html_body, subtype="html")

        ssl_context = ssl.create_default_context()

        with smtplib.SMTP_SSL(self.config.host, self.config.port, context=ssl_context) as server:
        
            server.login(self.config.username, self.config.password)
            server.send_message(message)
            print(f"Email sent successfully to {target}!")

    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """Convert a limited Markdown subset into HTML suitable for emails."""

        def format_inline(text: str) -> str:
            code_tokens: dict[str, str] = {}

            def replace_code(match: re.Match[str]) -> str:
                token = f"@@CODE{len(code_tokens)}@@"
                code_tokens[token] = html.escape(match.group(1))
                return token

            text_with_tokens = re.sub(r"`([^`]+)`", replace_code, text)
            escaped = html.escape(text_with_tokens)

            def replace_link(match: re.Match[str]) -> str:
                label = match.group(1)
                href = match.group(2)
                return f'<a href="{href}">{label}</a>'

            escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, escaped)
            escaped = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", escaped)
            escaped = re.sub(r"__(.+?)__", lambda m: f"<strong>{m.group(1)}</strong>", escaped)
            escaped = re.sub(
                r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", lambda m: f"<em>{m.group(1)}</em>", escaped
            )
            escaped = re.sub(
                r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", lambda m: f"<em>{m.group(1)}</em>", escaped
            )

            for token, value in code_tokens.items():
                escaped = escaped.replace(token, f"<code>{value}</code>")
            return escaped

        lines: list[str] = []
        list_stack: list[tuple[int, str]] = []

        def close_lists(target_indent: int) -> None:
            while list_stack and list_stack[-1][0] > target_indent:
                _, closing_tag = list_stack.pop()
                lines.append(f"</{closing_tag}>")

        for raw_line in markdown_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if not list_stack:
                    lines.append("<p></p>")
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            heading_match = re.match(r"(#{1,6})\s+(.*)", stripped)
            if heading_match:
                close_lists(-1)
                level = len(heading_match.group(1))
                content = format_inline(heading_match.group(2).strip())
                lines.append(f"<h{level}>{content}</h{level}>")
                continue

            bullet_match = re.match(r"[-*]\s+(.*)", stripped)
            numbered_match = re.match(r"\d+\.\s+(.*)", stripped)
            if bullet_match or numbered_match:
                list_type = "ul" if bullet_match else "ol"
                item_content = (bullet_match or numbered_match).group(1).strip()
                close_lists(indent)
                if not list_stack or list_stack[-1][0] != indent or list_stack[-1][1] != list_type:
                    while list_stack and list_stack[-1][0] == indent and list_stack[-1][1] != list_type:
                        _, closing_tag = list_stack.pop()
                        lines.append(f"</{closing_tag}>")
                    list_stack.append((indent, list_type))
                    lines.append(f"<{list_type}>")
                lines.append(f"<li>{format_inline(item_content)}</li>")
                continue

            close_lists(-1)
            lines.append(f"<p>{format_inline(stripped)}</p>")

        while list_stack:
            _, closing_tag = list_stack.pop()
            lines.append(f"</{closing_tag}>")

        return "\n".join(lines)


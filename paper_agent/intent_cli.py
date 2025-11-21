from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable

from .intent_agent import IntentAgent, IntentAgentError
from .intent_profiles import DEFAULT_PROFILE_NAME, IntentProfile, IntentProfileStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive intent profile authoring tool.")
    parser.add_argument(
        "--profile-name",
        default=DEFAULT_PROFILE_NAME,
        help="Name of the intent profile to create or update (default: %(default)s).",
    )
    parser.add_argument(
        "--description",
        help="Initial paragraph describing the research intent. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--config-dir",
        help="Custom directory for storing intent profiles (default: <repo>/config/intent_profiles).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the existing profile (if any) before applying new instructions.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    store = IntentProfileStore(args.config_dir)
    existing: IntentProfile | None = None
    if args.resume and store.exists(args.profile_name):
        try:
            existing = store.load(args.profile_name)
            logging.info("Loaded existing profile '%s' for editing.", args.profile_name)
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Failed to load existing profile '%s': %s", args.profile_name, exc)
    description = (args.description or "").strip()
    if not description:
        description = existing.description if existing else ""
    if not description:
        description = _prompt("Describe the research intent: ").strip()
    if not description:
        print("A non-empty description is required to build an intent profile.", file=sys.stderr)
        raise SystemExit(1)
    agent = IntentAgent()
    try:
        profile = agent.draft_profile(
            profile_name=args.profile_name,
            description=description,
            seed_profile=existing,
        )
    except IntentAgentError as exc:
        logging.error("LLM intent interpretation failed: %s", exc)
        raise SystemExit(1) from exc
    profile = _interactive_refinement(agent, profile)
    path = store.save(profile)
    print(f"Intent profile '{profile.name}' saved to {path}")


def _interactive_refinement(agent: IntentAgent, profile: IntentProfile) -> IntentProfile:
    confirmations = {"", "ok", "y", "yes", "done"}
    while True:
        _print_profile(profile)
        feedback = _prompt("\nPress Enter to accept, or describe adjustments: ").strip()
        if feedback.lower() in confirmations:
            return profile
        try:
            profile = agent.refine_profile(profile, feedback)
        except IntentAgentError as exc:
            logging.error("Unable to refine profile: %s", exc)
            continue


def _print_profile(profile: IntentProfile) -> None:
    divider = "=" * 60
    print(f"\n{divider}\nProfile: {profile.name}\n{divider}")
    print(f"Description: {profile.description}")
    if profile.notes:
        print(f"Notes: {profile.notes}")
    print(_format_section("Topics", profile.topics))
    print(_format_section("Keywords", profile.keywords))
    print(_format_section("Required Keywords", profile.required_keywords or ["(none)"]))


def _format_section(title: str, items: Iterable[str]) -> str:
    values = [f"- {item}" for item in items]
    return f"{title}:\n" + ("\n".join(values) if values else "- (empty)")


def _prompt(message: str) -> str:
    try:
        return input(message)
    except EOFError:
        return ""


if __name__ == "__main__":  # pragma: no cover
    main()



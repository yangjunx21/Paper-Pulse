#!/usr/bin/env python
"""
Utility script for inspecting how "Institutions" are populated in summaries.

It runs the regular pipeline (email delivery disabled) and prints, for each ranked
paper, the raw affiliations pulled during fetching plus the structured summary fields
parsed from the LLM output (Institutions/First Author/etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_agent.models import PipelineSettings
from paper_agent.pipeline import generate_recommendations

LOGGER = logging.getLogger("debug_institutions")


def _valid_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - simple input validation
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def _parse_structured_summary(summary: str) -> Dict[str, object]:
    """Minimal clone of pipeline logic so we can inspect structured fields."""
    result: Dict[str, object] = {
        "institutions": "",
        "first_author": "",
        "corresponding_author": "",
        "tldr": "",
        "key_contributions": [],
    }
    if not summary:
        return result
    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    collecting_keys = False
    for line in lines:
        lower = line.lower()
        if collecting_keys:
            if line[0].isdigit() and ". " in line:
                _, value = line.split(". ", 1)
                result["key_contributions"].append(value.strip())
                continue
            result["key_contributions"].append(line)
            continue
        if lower.startswith("key contributions"):
            collecting_keys = True
            continue
        if ":" not in line:
            continue
        label, value = [part.strip() for part in line.split(":", 1)]
        label_lower = label.lower()
        if label_lower.startswith("institutions"):
            result["institutions"] = value
        elif label_lower.startswith("first author"):
            result["first_author"] = value
        elif label_lower.startswith("corresponding author"):
            result["corresponding_author"] = value
        elif label_lower.startswith("tldr"):
            result["tldr"] = value
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Institutions/affiliations data for ranked papers.")
    parser.add_argument("--topics", nargs="+", help="Override default topics.")
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument("--date", type=_valid_date, help="Single target date (YYYY-MM-DD).")
    date_group.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        type=_valid_date,
        help="Inclusive date range (YYYY-MM-DD YYYY-MM-DD).",
    )
    parser.add_argument("--sources", nargs="+", help="Restrict to explicit sources (arxiv, huggingface_daily, neurips_2025).")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum candidates per topic.")
    parser.add_argument("--keywords-file", help="Optional keywords YAML.")
    parser.add_argument("--required-keywords", nargs="+", help="Required keyword list.")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity (default INFO).")
    parser.add_argument("--limit", type=int, default=None, help="Only display first N ranked papers.")
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_settings(args: argparse.Namespace) -> PipelineSettings:
    kwargs = {
        "max_results_per_topic": args.max_results,
        "send_email": False,
    }
    if args.topics:
        kwargs["topics"] = args.topics
    if args.sources:
        kwargs["sources"] = args.sources
    if args.keywords_file:
        kwargs["keywords_file"] = args.keywords_file
    if args.required_keywords:
        kwargs["required_keywords"] = args.required_keywords
    if args.date:
        kwargs["target_date"] = args.date
    elif args.date_range:
        kwargs["start_date"], kwargs["end_date"] = args.date_range
    return PipelineSettings(**kwargs)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    settings = _build_settings(args)

    LOGGER.info("Generating recommendations for inspection...")
    result = generate_recommendations(settings)
    to_show = result.papers[: args.limit] if args.limit else result.papers

    print(f"Total ranked papers: {len(result.papers)} (showing {len(to_show)})")
    print("-" * 80)
    for paper in to_show:
        structured = _parse_structured_summary(paper.summary or "")
        affiliations = paper.paper.affiliations or []
        print(f"{paper.rank}. {paper.paper.title}")
        print(f"   Raw affiliations : {affiliations or 'N/A'}")
        print(f"   Structured field : {structured.get('institutions') or 'N/A'}")
        print(f"   First author     : {structured.get('first_author') or 'Unknown'}")
        print(f"   Corresponding    : {structured.get('corresponding_author') or 'Unknown'}")
        print(f"   TLDR             : {structured.get('tldr') or 'Not provided'}")
        key_points = structured.get("key_contributions") or []
        if key_points:
            print("   Key contributions:")
            for idx, point in enumerate(key_points, start=1):
                print(f"      {idx}. {point}")
        else:
            print("   Key contributions: Not provided")
        print("-" * 80)

    print("Done. If 'Raw affiliations' is empty, consider augmenting fetchers or fetching PDF metadata.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


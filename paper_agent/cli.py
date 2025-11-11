from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from typing import List

from .models import AVAILABLE_SOURCES, PipelineSettings
from .pipeline import generate_gap_fill_digest, generate_recommendations


def _valid_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-powered paper recommendation agent")
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Override the default research focus (LLM Safety) with custom descriptors.",
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--date",
        type=_valid_date,
        help="Target arXiv submission date in YYYY-MM-DD (default: today).",
    )
    date_group.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START_DATE", "END_DATE"),
        type=_valid_date,
        help="Target arXiv submission date range in YYYY-MM-DD YYYY-MM-DD (inclusive).",
    )
    date_group.add_argument(
        "--gap-fill-week",
        nargs=2,
        metavar=("GAP_START", "GAP_END"),
        type=_valid_date,
        help="Run a weekly gap-fill digest using cached keyword-filtered papers from the inclusive date range.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=80,
        help="Max candidate papers to keep after keyword filtering (per run, default: 80).",
    )
    parser.add_argument("--send-email", action="store_true", help="Send email using SMTP config")
    parser.add_argument("--receiver", help="Override receiver email address")
    parser.add_argument("--output-json", help="Path to write pipeline result JSON")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.8,
        help="Minimum relevance score for including a paper in the final report (default: 0.8).",
    )
    parser.add_argument(
        "--fallback-report-limit",
        type=int,
        default=10,
        help="Number of top papers to show when no paper exceeds the threshold (default: 10).",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=4,
        help="Maximum number of parallel LLM requests (default: 4).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=AVAILABLE_SOURCES,
        help=(
            "Specify one or more data sources to query. "
            f"Choices: {', '.join(AVAILABLE_SOURCES)}. Default: {', '.join(PipelineSettings().sources)}."
        ),
    )
    parser.add_argument(
        "--keywords-file",
        help="Path to a YAML file that defines the keywords used for filtering and LLM prompts.",
    )
    parser.add_argument(
        "--required-keywords",
        nargs="+",
        help="Explicit list of keywords that must appear in every selected paper (overrides file setting).",
    )
    parser.add_argument(
        "--gap-fill-limit",
        type=int,
        help="Optional cap on the number of cached candidates to evaluate during a gap-fill run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings_kwargs = dict(
        max_results_per_topic=args.max_results,
        send_email=args.send_email,
        receiver_email=args.receiver,
    )
    if args.topics:
        settings_kwargs["topics"] = args.topics
    if args.date_range:
        start_date, end_date = args.date_range
        settings_kwargs["start_date"] = start_date
        settings_kwargs["end_date"] = end_date
    elif args.date is not None:
        settings_kwargs["target_date"] = args.date
    if args.sources:
        settings_kwargs["sources"] = args.sources
    if args.keywords_file:
        settings_kwargs["keywords_file"] = args.keywords_file
    if args.required_keywords:
        settings_kwargs["required_keywords"] = args.required_keywords
    settings_kwargs["relevance_threshold"] = args.relevance_threshold
    settings_kwargs["fallback_report_limit"] = args.fallback_report_limit
    settings_kwargs["llm_max_workers"] = args.llm_workers

    settings = PipelineSettings(**settings_kwargs)
    if args.gap_fill_week:
        week_start, week_end = args.gap_fill_week
        result = generate_gap_fill_digest(
            settings,
            week_start=week_start,
            week_end=week_end,
            max_candidates=args.gap_fill_limit,
        )
        print(f"Gap-fill subject: {result.email_subject}")
        print("Gap-fill report preview:")
    else:
        result = generate_recommendations(settings)
        print(f"Email subject: {result.email_subject}")
        print("Email preview:")
    print(result.email_body)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))
        print(f"Saved result to {args.output_json}")


if __name__ == "__main__":  # pragma: no cover
    main()


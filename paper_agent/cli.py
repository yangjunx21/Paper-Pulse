from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

from .models import AVAILABLE_SOURCES, PipelineSettings
from .pipeline import generate_gap_fill_digest, generate_recommendations
from .intent_profiles import (
    IntentProfile,
    IntentProfileNotFoundError,
    IntentProfileStore,
    apply_profile_defaults,
)

LOGGER = logging.getLogger(__name__)


def _valid_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer '{value}'.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


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
        "--summary-language",
        default="English",
        help="Language to use for structured LLM summaries (default: English).",
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
        "--intent-profile",
        help="Name of a stored intent profile to load (create via `python -m paper_agent.intent_cli`).",
    )
    parser.add_argument(
        "--intent-config-dir",
        help="Override the directory that stores intent profiles.",
    )
    parser.add_argument(
        "--skip-intent-profile",
        action="store_true",
        help="Disable automatic loading of saved intent profiles.",
    )
    parser.add_argument(
        "--enable-pdf-analysis",
        action="store_true",
        help="Download arXiv PDFs to enrich author affiliations and LLM full-text summaries.",
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
    parser.add_argument(
        "--repeat-every-days",
        type=_positive_int,
        help="Enable daemon mode and rerun the pipeline every N days.",
    )
    parser.add_argument(
        "--service-window-days",
        type=_positive_int,
        help="Size of the rolling date window (in days) when daemon mode is active. Defaults to N from --repeat-every-days.",
    )
    return parser.parse_args()


def _build_settings_kwargs(
    args: argparse.Namespace,
    *,
    date_range_override: Optional[Tuple[date, date]] = None,
) -> dict:
    settings_kwargs: dict = dict(
        max_results_per_topic=args.max_results,
        send_email=args.send_email,
        receiver_email=args.receiver,
    )
    if args.topics:
        settings_kwargs["topics"] = args.topics
    if args.sources:
        settings_kwargs["sources"] = args.sources
    if args.keywords_file:
        settings_kwargs["keywords_file"] = args.keywords_file
    if args.required_keywords:
        settings_kwargs["required_keywords"] = args.required_keywords
    settings_kwargs["enable_pdf_analysis"] = args.enable_pdf_analysis
    settings_kwargs["summary_language"] = args.summary_language
    settings_kwargs["relevance_threshold"] = args.relevance_threshold
    settings_kwargs["fallback_report_limit"] = args.fallback_report_limit
    settings_kwargs["llm_max_workers"] = args.llm_workers

    _maybe_apply_intent_profile(args, settings_kwargs)

    if date_range_override:
        start_date, end_date = date_range_override
        settings_kwargs["start_date"] = start_date
        settings_kwargs["end_date"] = end_date
    else:
        if args.date_range:
            start_date, end_date = args.date_range
            settings_kwargs["start_date"] = start_date
            settings_kwargs["end_date"] = end_date
        elif args.date is not None:
            settings_kwargs["target_date"] = args.date
    return settings_kwargs


def _maybe_apply_intent_profile(args: argparse.Namespace, settings_kwargs: dict) -> None:
    profile = _load_intent_profile(args)
    if not profile:
        return
    apply_profile_defaults(settings_kwargs, profile)
    LOGGER.info(
        "Loaded intent profile '%s' (topics=%d, keywords=%d, required=%d).",
        profile.name,
        len(profile.topics),
        len(profile.keywords),
        len(profile.required_keywords),
    )


def _load_intent_profile(args: argparse.Namespace) -> IntentProfile | None:
    if getattr(args, "skip_intent_profile", False):
        LOGGER.debug("Skipping intent profile loading because --skip-intent-profile is set.")
        return None
    profile_name = args.intent_profile
    if not profile_name:
        return None
    store = IntentProfileStore(args.intent_config_dir)
    try:
        return store.load(profile_name)
    except IntentProfileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def _run_pipeline(
    settings_kwargs: dict,
    *,
    gap_fill_week: Optional[Tuple[date, date]] = None,
    gap_fill_limit: Optional[int] = None,
):
    settings = PipelineSettings(**settings_kwargs)
    if gap_fill_week:
        week_start, week_end = gap_fill_week
        result = generate_gap_fill_digest(
            settings,
            week_start=week_start,
            week_end=week_end,
            max_candidates=gap_fill_limit,
        )
        subject_label = "Gap-fill subject"
        preview_label = "Gap-fill report preview"
    else:
        result = generate_recommendations(settings)
        subject_label = "Email subject"
        preview_label = "Email preview"
    return subject_label, preview_label, result


def _print_result(
    subject_label: str,
    preview_label: str,
    result,
    *,
    output_json: Optional[str] = None,
    header: Optional[str] = None,
) -> None:
    if header:
        print(header)
    print(f"{subject_label}: {result.email_subject}")
    print(f"{preview_label}:")
    print(result.email_body)
    if output_json:
        with open(output_json, "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))
        print(f"Saved result to {output_json}")


def run_once(args: argparse.Namespace) -> None:
    settings_kwargs = _build_settings_kwargs(args)
    gap_fill_week: Optional[Tuple[date, date]] = (
        tuple(args.gap_fill_week) if args.gap_fill_week else None
    )
    subject_label, preview_label, result = _run_pipeline(
        settings_kwargs,
        gap_fill_week=gap_fill_week,
        gap_fill_limit=args.gap_fill_limit,
    )
    _print_result(
        subject_label,
        preview_label,
        result,
        output_json=args.output_json,
    )


def run_service(args: argparse.Namespace) -> None:
    repeat_days = args.repeat_every_days
    if repeat_days is None:
        raise ValueError("repeat_every_days must be provided for service mode.")
    window_days = args.service_window_days or repeat_days
    repeat_interval = timedelta(days=repeat_days)
    next_run_time = datetime.now(timezone.utc)
    logging.info(
        "Starting daemon mode: interval=%s days, window=%s days. Next run at %s",
        repeat_days,
        window_days,
        next_run_time.isoformat(),
    )
    iteration = 0
    try:
        while True:
            now = datetime.now(timezone.utc)
            if now < next_run_time:
                sleep_seconds = max(1.0, (next_run_time - now).total_seconds())
                time.sleep(sleep_seconds)
                continue
            iteration += 1
            end_date = now.date()
            start_date = end_date - timedelta(days=window_days - 1)
            logging.info(
                "Daemon iteration %s: generating report for %s - %s",
                iteration,
                start_date,
                end_date,
            )
            settings_kwargs = _build_settings_kwargs(
                args, date_range_override=(start_date, end_date)
            )
            subject_label, preview_label, result = _run_pipeline(settings_kwargs)
            header = f"[daemon] iteration {iteration} ({start_date} to {end_date})"
            _print_result(
                subject_label,
                preview_label,
                result,
                output_json=args.output_json,
                header=header,
            )
            next_run_time = next_run_time + repeat_interval
            if next_run_time <= now:
                next_run_time = now + repeat_interval
            logging.info("Next daemon run scheduled at %s", next_run_time.isoformat())
    except KeyboardInterrupt:
        logging.info("Daemon mode interrupted; exiting.")


def main() -> None:
    args = parse_args()
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.repeat_every_days is not None:
        if args.gap_fill_week:
            raise SystemExit("Daemon mode does not support --gap-fill-week.")
        if args.service_window_days and args.service_window_days <= 0:
            raise SystemExit("--service-window-days must be positive.")
        if args.date_range or args.date:
            logging.warning("Daemon mode ignores --date/--date-range; using rolling window.")
        run_service(args)
    else:
        if args.service_window_days is not None:
            raise SystemExit("--service-window-days requires --repeat-every-days.")
        run_once(args)


if __name__ == "__main__":  # pragma: no cover
    main()


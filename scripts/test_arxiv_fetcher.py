from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_agent.fetchers.arxiv_fetcher import ArxivFetcher

LOGGER = logging.getLogger("arxiv_fetcher_test")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_date(value: Optional[str]) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD.") from exc


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a manual fetch against the arXiv API.")
    parser.add_argument("--target-date", type=_parse_date, help="Target publication date (YYYY-MM-DD).")
    parser.add_argument("--start-date", type=_parse_date, help="Start of publication date range (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=_parse_date, help="End of publication date range (YYYY-MM-DD).")
    parser.add_argument("--max-results", type=int, default=None, help="Maximum number of results to retrieve.")
    parser.add_argument("--page-size", type=int, default=200, help="Number of results to request per API page.")
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        help="Restrict results to the given arXiv category (can be provided multiple times).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def _print_papers(papers: Iterable) -> None:
    for paper in papers:
        published = paper.published.isoformat()
        categories = ", ".join(paper.categories) if paper.categories else "-"
        authors = ", ".join(paper.authors) if paper.authors else "-"
        print("=" * 80)
        print(f"Title     : {paper.title}")
        print(f"ID        : {paper.id}")
        print(f"Published : {published}")
        print(f"Authors   : {authors}")
        print(f"Categories: {categories}")
        print(f"Link      : {paper.link}")
        print(f"Summary   : {paper.summary.strip() or '-'}")
    print("=" * 80)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    fetcher = ArxivFetcher(page_size=args.page_size, categories=args.categories)

    if args.start_date and not args.end_date or args.end_date and not args.start_date:
        parser.error("Both --start-date and --end-date are required when using a date range.")

    if args.target_date and (args.start_date or args.end_date):
        parser.error("--target-date cannot be combined with --start-date/--end-date.")

    try:
        papers = fetcher.fetch(
            target_date=args.target_date,
            start_date=args.start_date,
            end_date=args.end_date,
            max_results=args.max_results,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Fetching failed: %s", exc)
        return 1

    LOGGER.info("Fetched %d paper(s) from arXiv.", len(papers))
    _print_papers(papers)

    return 0


if __name__ == "__main__":
    sys.exit(main())


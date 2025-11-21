from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .cache import CacheManager
from .config import get_optional_email_config
from .fetchers import PaperFetcher, get_fetcher
from .llm import LLMChatRequest, LLMClient, LLMMessage
from .llm.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    build_classification_system_prompt,
    CLASSIFICATION_USER_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
    AFFILIATION_SYSTEM_PROMPT,
    AFFILIATION_USER_PROMPT_TEMPLATE,
)
from .llm.utils import prepare_llm_json_content
from .mailer import EmailClient
from .models import ClassifiedPaper, PipelineResult, PipelineSettings, RankedPaper, RawPaper
from .parsers.pdf_utils import (
    download_pdf,
    extract_affiliation_snippet_from_pdf_bytes,
    extract_text_from_pdf_bytes,
    infer_affiliations_from_text,
)
from .rankers import HybridRanker
from .keywords import resolve_keywords
from .progress import iter_with_progress

LOGGER = logging.getLogger(__name__)

@dataclass
class Layer1FilterOutcome:
    retained: List[RawPaper]
    removed_by_category: List[RawPaper]
    removed_by_keywords: List[RawPaper]
    removed_by_required: List[RawPaper]


def generate_recommendations(settings: PipelineSettings) -> PipelineResult:
    email_config = get_optional_email_config()
    if settings.send_email and not email_config:
        raise ValueError("Send email requested but email config environment variables are missing.")
    should_send_email = bool(settings.send_email and email_config)
    if settings.start_date and settings.end_date:
        date_descriptor = f"{settings.start_date} to {settings.end_date}"
    else:
        date_descriptor = settings.target_date.isoformat()
    source_descriptor = ", ".join(settings.sources)
    focus = "; ".join(settings.topics) if settings.topics else "LLM Safety"
    total_steps = 7
    if settings.enable_pdf_analysis:
        total_steps += 1
    if should_send_email:
        total_steps += 1
    current_step = 0

    def log_step(description: str) -> int:
        nonlocal current_step
        current_step += 1
        LOGGER.info("[Step %d/%d] %s", current_step, total_steps, description)
        return current_step

    report_date = settings.target_date or settings.end_date or settings.start_date or datetime.now(timezone.utc).date()
    LOGGER.info(
        "Starting paper recommendation pipeline for %s (sources: %s, focus: %s, email=%s).",
        date_descriptor,
        source_descriptor,
        focus,
        "enabled" if should_send_email else "disabled",
    )
    cache = CacheManager()
    init_step_index = log_step("Initializing data sources and clients")
    llm = LLMClient()
    raw_papers = cache.load_raw_papers(settings)
    fetchers: List[PaperFetcher] = []
    if raw_papers is None:
        fetchers = _instantiate_fetchers(settings.sources)
        if not fetchers:
            raise ValueError(
                f"No valid fetchers available for requested sources: {', '.join(settings.sources)}."
            )
        LOGGER.info(
            "[Step %d/%d] Prepared %d data sources: %s",
            init_step_index,
            total_steps,
            len(fetchers),
            ", ".join(fetcher.source_name for fetcher in fetchers),
        )
    else:
        LOGGER.info(
            "[Step %d/%d] Cache hit, skipping data source initialization (sources: %s).",
            init_step_index,
            total_steps,
            source_descriptor,
        )

    raw_key: Optional[str] = None
    if raw_papers is None:
        fetch_step_index = log_step("Fetching raw paper data")
        fetch_kwargs = {
            "target_date": settings.target_date,
            "start_date": settings.start_date,
            "end_date": settings.end_date,
            "max_results": None,
        }
        aggregated: Dict[str, RawPaper] = {}
        total_collected = 0
        for idx, fetcher in enumerate(
            iter_with_progress(
                fetchers,
                description="Fetching from sources",
                total=len(fetchers),
            ),
            start=1,
        ):
            LOGGER.info(
                "[Step %d/%d] Fetching progress %d/%d -> source %s",
                fetch_step_index,
                total_steps,
                idx,
                len(fetchers),
                fetcher.source_name,
            )
            try:
                fetched = fetcher.fetch(**fetch_kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("Failed to fetch papers from source '%s': %s", fetcher.source_name, exc, exc_info=exc)
                continue
            LOGGER.info("Source %s returned %d papers.", fetcher.source_name, len(fetched))
            for paper in iter_with_progress(
                fetched,
                description=f"Processing {fetcher.source_name}",
                total=len(fetched),
            ):
                total_collected += 1
                existing = aggregated.get(paper.id)
                if existing:
                    aggregated[paper.id] = _merge_raw_papers(existing, paper)
                else:
                    aggregated[paper.id] = paper
        raw_papers = list(aggregated.values())
        if not raw_papers:
            raise ValueError(
                f"No papers fetched for {date_descriptor} from sources: {', '.join(settings.sources)}."
            )
        if total_collected != len(raw_papers):
            LOGGER.info(
                "Collected %d unique papers after deduplication (total fetched=%d).",
                len(raw_papers),
                total_collected,
            )
        raw_key = cache.store_raw_papers(settings, raw_papers)
        LOGGER.info(
            "Cached %d raw papers for %s (key=%s).",
            len(raw_papers),
            date_descriptor,
            raw_key[:8],
        )
    else:
        raw_key = cache.raw_key(settings)
        fetch_step_index = log_step("Loading cached raw paper data")
        LOGGER.info(
            "[Step %d/%d] Loaded %d raw papers from cache (key=%s).",
            fetch_step_index,
            total_steps,
            len(raw_papers),
            raw_key[:8],
        )

    keyword_config = resolve_keywords(
        settings.keywords,
        settings.required_keywords,
        keyword_file=settings.keywords_file,
    )
    keywords = keyword_config.keywords
    required_keywords = keyword_config.required_keywords
    LOGGER.info(
        "Using %d keywords and %d required keywords for Layer 1 filtering and prompts.",
        len(keywords),
        len(required_keywords),
    )
    filter_step_index = log_step("Applying Layer 1 filters")
    LOGGER.info(
        "Fetched %d papers for %s from sources %s. Applying Layer 1 category and keyword filters.",
        len(raw_papers),
        date_descriptor,
        source_descriptor,
    )
    layer1_candidates = cache.load_layer1_candidates(
        settings,
        raw_papers,
        keywords=keywords,
        required_keywords=required_keywords,
    )
    filter_outcome: Optional[Layer1FilterOutcome] = None
    if layer1_candidates is None:
        filter_outcome = apply_layer1_filters_with_trace(raw_papers, keywords, required_keywords)
        layer1_candidates = filter_outcome.retained
        cache.store_layer1_result(
            settings,
            raw_key=raw_key or cache.raw_key(settings),
            raw_papers=raw_papers,
            retained=filter_outcome.retained,
            removed_by_category=filter_outcome.removed_by_category,
            removed_by_keywords=filter_outcome.removed_by_keywords,
            removed_by_required=filter_outcome.removed_by_required,
            keywords=keywords,
            required_keywords=required_keywords,
        )
        LOGGER.info(
            "[Step %d/%d] Layer 1 retention=%d, removed(category=%d, keyword=%d, required=%d).",
            filter_step_index,
            total_steps,
            len(filter_outcome.retained),
            len(filter_outcome.removed_by_category),
            len(filter_outcome.removed_by_keywords),
            len(filter_outcome.removed_by_required),
        )
    else:
        LOGGER.info(
            "[Step %d/%d] Layer 1 candidates loaded from cache with %d retained papers.",
            filter_step_index,
            total_steps,
            len(layer1_candidates),
        )

    if settings.max_results_per_topic > 0:
        layer1_candidates = layer1_candidates[: settings.max_results_per_topic]
    if not layer1_candidates:
        raise ValueError("No papers passed Layer 1 filtering for the requested date range.")
    LOGGER.info(
        "Layer 1 retained %d papers (limit=%s). Proceeding to Layer 2 LLM scoring.",
        len(layer1_candidates),
        settings.max_results_per_topic if settings.max_results_per_topic > 0 else "none",
    )

    log_step("Performing LLM classification")
    classification_system_prompt = build_classification_system_prompt(keywords, required_keywords)
    classified = classify_with_llm(
        llm,
        layer1_candidates,
        focus,
        keywords,
        required_keywords,
        classification_system_prompt,
        max_workers=settings.llm_max_workers,
        cache=cache,
    )
    if not classified:
        raise ValueError("LLM classification returned no valid papers.")

    log_step("Ranking and selecting core papers")
    ranked_all = HybridRanker().rank(classified)
    relevant_ranked = [
        paper
        for paper in ranked_all
        if paper.is_relevant or paper.relevance_score >= settings.relevance_threshold
    ]
    if not relevant_ranked:
        LOGGER.warning(
            "No papers met the relevance threshold %.2f; using top %d LLM-scored papers instead.",
            settings.relevance_threshold,
            settings.fallback_report_limit,
        )
        relevant_ranked = ranked_all[: settings.fallback_report_limit]
    relevant_ranked = [
        paper.model_copy(update={"rank": idx})
        for idx, paper in enumerate(relevant_ranked, start=1)
    ]
    if settings.enable_pdf_analysis:
        pdf_step_index = log_step("Enriching ranked papers with PDF full text")
        arxiv_candidates = [paper.paper for paper in relevant_ranked if paper.paper.source.lower() == "arxiv"]
        if arxiv_candidates:
            enriched_count, affiliation_updates = _enrich_papers_with_pdf_content(arxiv_candidates)
            if enriched_count or affiliation_updates:
                raw_key = cache.store_raw_papers(settings, raw_papers)
                LOGGER.info(
                    "[Step %d/%d] PDF enrichment complete: %d texts updated, %d affiliation lists enhanced (cache=%s).",
                    pdf_step_index,
                    total_steps,
                    enriched_count,
                    affiliation_updates,
                    raw_key[:8],
                )
            else:
                LOGGER.info(
                    "[Step %d/%d] Ranked papers already had PDF content; no updates applied.",
                    pdf_step_index,
                    total_steps,
                )
        else:
            LOGGER.info(
                "[Step %d/%d] No ranked arXiv papers required PDF enrichment.",
                pdf_step_index,
                total_steps,
            )
    relevant_ranked, missing_summaries = _apply_cached_summaries(cache, relevant_ranked)
    summary_step_index = log_step("Generating LLM summaries")
    if not settings.enable_pdf_analysis:
        message = (
            "Skipping LLM summaries because PDF reading is disabled (missing summaries remain)."
            if missing_summaries > 0
            else "All summaries sourced from cache; PDF reading is disabled so no new LLM calls."
        )
        LOGGER.info(
            "[Step %d/%d] %s",
            summary_step_index,
            total_steps,
            message,
        )
    elif missing_summaries > 0:
        LOGGER.info(
            "[Step %d/%d] Will generate or update summaries for %d papers, remaining summaries from cache.",
            summary_step_index,
            total_steps,
            missing_summaries,
        )
        relevant_ranked = summarize_ranked_papers(
            llm,
            relevant_ranked,
            tldr_language=settings.summary_language,
        )
        for ranked_paper in relevant_ranked:
            if cache and ranked_paper.summary:
                cache.store_summary(ranked_paper.paper, ranked_paper.summary)
    else:
        LOGGER.info(
            "[Step %d/%d] All summaries loaded from cache, skipping LLM calls.",
            summary_step_index,
            total_steps,
        )
    log_step("Building daily report content")
    email_subject, email_body = build_daily_report(
        relevant_ranked,
        focus,
        date_descriptor,
        source_descriptor,
        include_llm_analysis=settings.enable_pdf_analysis,
    )
    _persist_local_digest(email_subject, email_body)

    external_outputs: Dict[str, Any] = {}
    if should_send_email and email_config:
        log_step("Sending email notification")
        EmailClient(email_config).send_markdown_email(
            subject=email_subject,
            body_markdown=email_body,
            receiver=settings.receiver_email,
        )
    LOGGER.info(
        "Pipeline completed for %s. Final report contains %d papers.",
        date_descriptor,
        len(relevant_ranked),
    )

    return PipelineResult(
        settings=settings,
        papers=relevant_ranked,
        email_subject=email_subject,
        email_body=email_body,
        external_outputs=external_outputs,
    )


def generate_gap_fill_digest(
    settings: PipelineSettings,
    *,
    week_start: date,
    week_end: date,
    max_candidates: Optional[int] = None,
) -> PipelineResult:
    if week_start > week_end:
        raise ValueError("week_start must be on or before week_end.")
    email_config = get_optional_email_config()
    if settings.send_email and not email_config:
        raise ValueError("Send email requested but email config environment variables are missing.")
    should_send_email = bool(settings.send_email and email_config)
    cache = CacheManager()
    llm = LLMClient()
    focus = "; ".join(settings.topics) if settings.topics else "LLM Safety"
    source_descriptor = ", ".join(settings.sources)
    LOGGER.info(
        "Starting weekly gap-fill run for %s to %s (focus=%s).",
        week_start.isoformat(),
        week_end.isoformat(),
        focus,
    )
    candidates = cache.get_keyword_filtered_papers(start_date=week_start, end_date=week_end)
    if not candidates:
        raise ValueError("No cached keyword-filtered papers found for the requested week.")
    if max_candidates:
        candidates = candidates[:max_candidates]
    unique_candidates: Dict[str, RawPaper] = {}
    for paper in iter_with_progress(
        candidates,
        description="Preparing weekly candidates",
        total=len(candidates),
    ):
        unique_candidates.setdefault(paper.id, paper)
    candidate_list = list(unique_candidates.values())
    LOGGER.info(
        "Gap-fill classification will evaluate %d previously filtered papers.",
        len(candidate_list),
    )
    keyword_config = resolve_keywords(
        settings.keywords,
        settings.required_keywords,
        keyword_file=settings.keywords_file,
    )
    classification_system_prompt = build_classification_system_prompt(
        keyword_config.keywords,
        keyword_config.required_keywords,
    )
    classified = classify_with_llm(
        llm,
        candidate_list,
        focus,
        keyword_config.keywords,
        keyword_config.required_keywords,
        classification_system_prompt,
        max_workers=settings.llm_max_workers,
        cache=cache,
    )
    if not classified:
        raise ValueError("Weekly gap-fill classification returned no valid papers.")
    ranked_all = HybridRanker().rank(classified)
    relevant_ranked = [
        paper
        for paper in ranked_all
        if paper.is_relevant or paper.relevance_score >= settings.relevance_threshold
    ]
    if not relevant_ranked:
        LOGGER.warning(
            "Weekly gap-fill found no papers above threshold %.2f; returning top %d candidates instead.",
            settings.relevance_threshold,
            settings.fallback_report_limit,
        )
        relevant_ranked = ranked_all[: settings.fallback_report_limit]
    relevant_ranked = [
        paper.model_copy(update={"rank": idx})
        for idx, paper in enumerate(relevant_ranked, start=1)
    ]
    relevant_ranked, missing_summaries = _apply_cached_summaries(cache, relevant_ranked)
    if settings.enable_pdf_analysis and missing_summaries > 0:
        LOGGER.info(
            "Generating new summaries for %d gap-fill papers.",
            missing_summaries,
        )
        relevant_ranked = summarize_ranked_papers(
            llm,
            relevant_ranked,
            tldr_language=settings.summary_language,
        )
        for ranked_paper in relevant_ranked:
            if ranked_paper.summary:
                cache.store_summary(ranked_paper.paper, ranked_paper.summary)
    elif missing_summaries > 0 and not settings.enable_pdf_analysis:
        LOGGER.info(
            "Skipping LLM summaries for %d gap-fill papers because PDF reading is disabled.",
            missing_summaries,
        )
    date_descriptor = f"{week_start.isoformat()} to {week_end.isoformat()} (keyword gap-fill)"
    email_subject, email_body = build_daily_report(
        relevant_ranked,
        focus,
        date_descriptor,
        source_descriptor,
        include_llm_analysis=settings.enable_pdf_analysis,
    )
    _persist_local_digest(email_subject, email_body)
    external_outputs: Dict[str, Any] = {}
    if should_send_email and email_config:
        EmailClient(email_config).send_markdown_email(
            subject=email_subject,
            body_markdown=email_body,
            receiver=settings.receiver_email,
        )
    LOGGER.info(
        "Weekly gap-fill completed with %d highlighted papers.",
        len(relevant_ranked),
    )
    return PipelineResult(
        settings=settings,
        papers=relevant_ranked,
        email_subject=email_subject,
        email_body=email_body,
        external_outputs=external_outputs,
    )


def _persist_local_digest(
    subject: str,
    body_markdown: str,
    *,
    directory: Optional[Path] = None,
) -> Path:
    target_dir = directory or (Path.cwd() / "reports")
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_subject = re.sub(r"[^A-Za-z0-9_-]+", "-", subject).strip("-")
    if not safe_subject:
        safe_subject = "digest"
    filename = f"{timestamp}_{safe_subject}.md"
    filepath = target_dir / filename
    content = f"# {subject}\n\n{body_markdown.strip()}\n"
    filepath.write_text(content, encoding="utf-8")
    LOGGER.info("Saved local digest copy to %s", filepath)
    return filepath


def _instantiate_fetchers(source_names: Sequence[str]) -> List[PaperFetcher]:
    fetchers: List[PaperFetcher] = []
    for source in source_names:
        try:
            fetchers.append(get_fetcher(source))
        except ValueError as exc:
            LOGGER.error("Unable to initialize fetcher for source '%s': %s", source, exc)
    return fetchers


AFFILIATION_MIN_COUNT = 1


def _enrich_papers_with_pdf_content(papers: Sequence[RawPaper]) -> Tuple[int, int]:
    enriched = 0
    affiliation_updates = 0
    llm = LLMClient()  # Instantiate LLM for affiliation extraction

    for paper in iter_with_progress(
        papers,
        description="PDF enrichment",
        total=len(papers),
    ):
        if paper.source.lower() != "arxiv":
            continue
        has_full_text = _has_meaningful_full_text(paper)
        needs_full_text = not has_full_text
        
        # Check if affiliations are truly present (ignoring "Unknown", "None", etc.)
        current_affiliations = paper.affiliations or []
        valid_affiliations = [
            aff for aff in current_affiliations 
            if aff and aff.lower() not in {"unknown", "none", "not provided", "null"}
        ]
        needs_affiliations = len(valid_affiliations) < AFFILIATION_MIN_COUNT
        
        # Case 1: We already have full text but need affiliations
        if has_full_text and needs_affiliations:
            full_text_content = getattr(paper, "full_text", "") or ""
            inferred_from_existing = _extract_affiliations_with_llm(llm, full_text_content)
            if inferred_from_existing and inferred_from_existing != paper.affiliations:
                paper.affiliations = inferred_from_existing
                affiliation_updates += 1
            continue
            
        # Case 2: We need to fetch PDF
        if not needs_full_text and not needs_affiliations:
            continue
        pdf_url = _resolve_pdf_url(paper)
        if not pdf_url:
            continue
        pdf_payload = download_pdf(pdf_url)
        if not pdf_payload:
            continue
            
        # Try to get full text if needed
        full_text = ""
        if needs_full_text:
            full_text = extract_text_from_pdf_bytes(pdf_payload)
            if full_text:
                paper.full_text = full_text
                enriched += 1
        
        # If we just fetched full text (or have it), and need affiliations, use LLM
        affiliation_source = full_text
        if not affiliation_source and needs_affiliations:
            # If full extraction failed or wasn't requested, try snippet extraction
            snippet_text = extract_affiliation_snippet_from_pdf_bytes(pdf_payload)
            affiliation_source = snippet_text

        if needs_affiliations and affiliation_source:
            inferred_affiliations = _extract_affiliations_with_llm(llm, affiliation_source)
            if inferred_affiliations and inferred_affiliations != paper.affiliations:
                paper.affiliations = inferred_affiliations
                affiliation_updates += 1
    return enriched, affiliation_updates


def _extract_affiliations_with_llm(llm: LLMClient, text: str) -> List[str]:
    """Extracts affiliations using LLM from the provided text."""
    if not text:
        return []
    
    # Take first ~500 words or ~3000 chars to stay within reasonable context
    # Usually affiliations are at the start.
    truncated_text = text[:3500] 
    
    payload = AFFILIATION_USER_PROMPT_TEMPLATE.format(text=truncated_text)
    messages = [
        LLMMessage(role="system", content=AFFILIATION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=payload),
    ]
    
    try:
        response = llm.chat_completion(messages)
        content = (response.content or "").strip()
        if not content or content.lower() == "unknown":
            return []
        
        # Split by comma and clean up
        affiliations = [aff.strip() for aff in content.split(",") if aff.strip()]
        return affiliations
    except Exception as e:
        LOGGER.warning("Failed to extract affiliations with LLM: %s", e)
        return []


def _has_meaningful_full_text(paper: RawPaper, *, min_chars: int = 600) -> bool:
    content = getattr(paper, "full_text", None)
    if not isinstance(content, str):
        return False
    return len(content.strip()) >= min_chars


def _resolve_pdf_url(paper: RawPaper) -> Optional[str]:
    pdf_url = getattr(paper, "pdf_url", None)
    if isinstance(pdf_url, str) and pdf_url.strip():
        return pdf_url
    link = str(paper.link or "")
    if "arxiv.org" in link and "/abs/" in link:
        pdf_url = link.replace("/abs/", "/pdf/")
        if not pdf_url.endswith(".pdf"):
            pdf_url = f"{pdf_url}.pdf"
        return pdf_url
    return None


def _merge_raw_papers(primary: RawPaper, secondary: RawPaper) -> RawPaper:
    sources = {primary.source}
    sources.add(secondary.source)
    categories = list(dict.fromkeys([*primary.categories, *secondary.categories]))
    summary = primary.summary
    if secondary.summary and (not summary or len(secondary.summary) > len(summary)):
        summary = secondary.summary
    authors = primary.authors or secondary.authors
    link = primary.link or secondary.link
    published = primary.published
    if secondary.published and secondary.published > primary.published:
        published = secondary.published
    LOGGER.debug(
        "Merging duplicate paper %s from sources %s",
        primary.id,
        ", ".join(sorted(sources)),
    )
    return primary.model_copy(
        update={
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "link": link,
            "published": published,
            "source": primary.source,
        }
    )


def apply_layer1_filters(
    papers: Iterable[RawPaper],
    keywords: Iterable[str],
    required_keywords: Iterable[str],
) -> List[RawPaper]:
    outcome = apply_layer1_filters_with_trace(papers, keywords, required_keywords)
    return outcome.retained


def apply_layer1_filters_with_trace(
    papers: Iterable[RawPaper],
    keywords: Iterable[str],
    required_keywords: Iterable[str],
) -> Layer1FilterOutcome:
    paper_list = list(papers)
    LOGGER.debug("Layer 1 starting with %d papers.", len(paper_list))
    categorized = filter_papers_by_categories(paper_list)
    keyword_filtered = filter_papers_by_keywords(categorized, keywords)
    required_filtered = filter_papers_by_required_keywords(keyword_filtered, required_keywords)
    LOGGER.debug(
        "Layer 1 completed: %d papers after categories, %d after keywords, %d after required keywords.",
        len(categorized),
        len(keyword_filtered),
        len(required_filtered),
    )
    categorized_ids = {paper.id for paper in categorized}
    keyword_ids = {paper.id for paper in keyword_filtered}
    required_ids = {paper.id for paper in required_filtered}
    removed_by_category = [paper for paper in paper_list if paper.id not in categorized_ids]
    removed_by_keywords = [paper for paper in categorized if paper.id not in keyword_ids]
    removed_by_required = [paper for paper in keyword_filtered if paper.id not in required_ids]
    return Layer1FilterOutcome(
        retained=required_filtered,
        removed_by_category=removed_by_category,
        removed_by_keywords=removed_by_keywords,
        removed_by_required=removed_by_required,
    )


def filter_papers_by_categories(papers: Iterable[RawPaper]) -> List[RawPaper]:
    paper_list = list(papers)
    LOGGER.debug(
        "Category filtering is handled by individual fetchers; returning %d paper(s) unchanged.",
        len(paper_list),
    )
    return paper_list


def filter_papers_by_keywords(papers: Iterable[RawPaper], keywords: Iterable[str]) -> List[RawPaper]:
    keyword_list = [keyword.strip().lower() for keyword in keywords if str(keyword).strip()]
    LOGGER.info("Applying keyword filter with %d keywords.", len(keyword_list))
    paper_list = list(papers)
    if not keyword_list:
        LOGGER.debug("No keywords provided; returning all %d papers.", len(paper_list))
        return paper_list
    filtered: List[RawPaper] = []
    for paper in iter_with_progress(
        paper_list,
        description="Keyword filtering",
        total=len(paper_list),
    ):
        haystack = f"{paper.title} {paper.summary}".lower()
        if any(keyword in haystack for keyword in keyword_list):
            filtered.append(paper)
    LOGGER.info("Keyword filter retained %d papers out of %d.", len(filtered), len(paper_list))
    return filtered


def filter_papers_by_required_keywords(
    papers: Iterable[RawPaper],
    required_keywords: Iterable[str],
) -> List[RawPaper]:
    required_list = [keyword.strip().lower() for keyword in required_keywords if str(keyword).strip()]
    paper_list = list(papers)
    if not required_list:
        LOGGER.debug("No required keywords provided; returning %d papers after optional filter.", len(paper_list))
        return paper_list
    LOGGER.info("Applying mandatory keyword filter with %d required keywords.", len(required_list))
    filtered: List[RawPaper] = []
    for paper in iter_with_progress(
        paper_list,
        description="Required keyword filtering",
        total=len(paper_list),
    ):
        haystack = f"{paper.title} {paper.summary}".lower()
        if any(required in haystack for required in required_list):
            filtered.append(paper)
    LOGGER.info(
        "Mandatory keyword filter retained %d papers out of %d.",
        len(filtered),
        len(paper_list),
    )
    return filtered


def classify_with_llm(
    llm: LLMClient,
    papers: Iterable[RawPaper],
    focus: str,
    keywords: Sequence[str],
    required_keywords: Sequence[str],
    system_prompt: str,
    *,
    max_workers: int = 4,
    cache: CacheManager | None = None,
) -> List[ClassifiedPaper]:
    paper_list = list(papers)
    if not paper_list:
        LOGGER.info("No papers provided for LLM classification.")
        return []

    result_map: Dict[str, ClassifiedPaper] = {}
    pending_requests: List[LLMChatRequest] = []
    pending_papers: List[RawPaper] = []
    cached_exact = 0
    cached_fallback = 0
    keywords_text = ", ".join(keywords) if keywords else "Not provided"
    required_keywords_text = ", ".join(required_keywords) if required_keywords else "None"

    for paper in iter_with_progress(
        paper_list,
        description="Preparing LLM classification payloads",
        total=len(paper_list),
    ):
        cached_result: Optional[ClassifiedPaper] = None
        exact_match = False
        if cache:
            cached_result, exact_match = cache.load_classification(
                paper,
                focus=focus,
                keywords=keywords,
                required_keywords=required_keywords,
                system_prompt=system_prompt,
                allow_cross_context=True,
            )
        if cached_result:
            result_map[paper.id] = cached_result
            if exact_match:
                cached_exact += 1
            else:
                cached_fallback += 1
            continue

        payload = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
            focus=focus,
            keywords=keywords_text,
            required_keywords=required_keywords_text,
            title=paper.title.strip(),
            abstract=paper.summary.strip(),
            categories=", ".join(paper.categories) if paper.categories else "unknown",
        )
        LOGGER.debug(
            "Layer 2 prompt for paper %s (truncated to 400 chars): %s",
            paper.id,
            payload[:400],
        )
        messages = [
            LLMMessage(role="system", content=system_prompt or CLASSIFICATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=payload),
        ]
        pending_requests.append(
            LLMChatRequest(
                messages=messages,
                temperature=0.0,
                metadata={"paper_id": paper.id},
            )
        )
        pending_papers.append(paper)

    fresh_success = 0
    if pending_requests:
        LOGGER.info(
            "Submitting %d papers for LLM classification (max_workers=%d, cache_exact=%d, cache_cross_topic=%d).",
            len(pending_requests),
            max_workers,
            cached_exact,
            cached_fallback,
        )
        batched_results = llm.chat_completion_batch(
            pending_requests,
            max_workers=max_workers,
            allow_errors=True,
        )

        progress_results = iter_with_progress(
            batched_results,
            description="Processing LLM classification responses",
            total=len(batched_results),
        )
        for paper, result in zip(pending_papers, progress_results):
            if result.error:
                LOGGER.warning(
                    "Skipping paper %s due to LLM error: %s",
                    paper.id,
                    result.error,
                )
                continue
            response = result.output
            if response is None:
                LOGGER.warning("LLM returned no content for paper %s; skipping.", paper.id)
                continue
            try:
                parsed_content = prepare_llm_json_content(response.content, context="classification")
                data = json.loads(parsed_content)
            except (json.JSONDecodeError, ValueError) as exc:
                LOGGER.warning(
                    "Skipping paper %s due to JSON parse error: %s. Raw response: %s",
                    paper.id,
                    exc,
                    response.content,
                )
                continue
            is_relevant = bool(data.get("is_relevant"))
            try:
                relevance_score = float(data.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                relevance_score = 0.0
            relevance_score = max(0.0, min(1.0, relevance_score))
            reasoning = data.get("reasoning")
            main_topic = data.get("main_topic")
            classified = ClassifiedPaper(
                paper=paper,
                is_relevant=is_relevant,
                relevance_score=relevance_score,
                main_topic=main_topic,
                reasoning=reasoning,
            )
            result_map[paper.id] = classified
            fresh_success += 1
            LOGGER.debug(
                "Layer 2 result: %s relevance=%.2f main_topic=%s",
                paper.id,
                relevance_score,
                main_topic,
            )
            if cache:
                cache.store_classification(
                    classified,
                    focus=focus,
                    keywords=keywords,
                    required_keywords=required_keywords,
                    system_prompt=system_prompt,
                )
    else:
        LOGGER.info(
            "Skipped LLM classification call; all %d papers served from cache (exact=%d, cross_topic=%d).",
            len(paper_list),
            cached_exact,
            cached_fallback,
        )

    results: List[ClassifiedPaper] = []
    for paper in iter_with_progress(
        paper_list,
        description="Collecting classification results",
        total=len(paper_list),
    ):
        classified = result_map.get(paper.id)
        if classified:
            results.append(classified)
        else:
            LOGGER.warning("No classification result retained for paper %s; it will be skipped.", paper.id)

    LOGGER.info(
        "LLM classification produced %d scored papers (cache_exact=%d, cache_cross_topic=%d, fresh=%d).",
        len(results),
        cached_exact,
        cached_fallback,
        fresh_success,
    )
    if cached_fallback > 0:
        LOGGER.info(
            "Reused %d cached classifications computed under different topic contexts to save LLM calls.",
            cached_fallback,
        )
    return results


def build_daily_report(
    ranked: Sequence[RankedPaper],
    focus: str,
    date_descriptor: str,
    sources_descriptor: str,
    *,
    include_llm_analysis: bool,
) -> tuple[str, str]:
    def _format_authors(authors: Sequence[str], max_display: int = 10, tail_keep: int = 3) -> str:
        names = [str(author).strip() for author in authors if str(author).strip()]
        if not names:
            return "Unknown"
        if len(names) <= max_display:
            return ", ".join(names)
        tail = min(tail_keep, max_display - 1)
        head = max_display - tail - 1
        if head <= 0:
            head = max(1, max_display - tail)
        head_section = names[:head]
        tail_section = names[-tail:] if tail > 0 else []
        return ", ".join([*head_section, "…", *tail_section])

    def _resolve_institutions(summary_value: str, affiliations: Sequence[str]) -> str:
        if summary_value and summary_value.strip() and summary_value.strip().lower() != "unknown":
            return summary_value.strip()
        unique: List[str] = []
        for affiliation in affiliations:
            text = str(affiliation).strip()
            if text and text not in unique:
                unique.append(text)
            if len(unique) >= 10:
                break
        return ", ".join(unique) if unique else "Unknown"

    def _extract_institutions_from_summary_text(
        summary_text: str,
        affiliations: Sequence[str],
    ) -> str:
        fallback = _resolve_institutions("", affiliations)
        if not summary_text:
            return fallback
        # Support both old emoji style and new markdown bullet style
        match = re.search(r"^(?:🏛️|- \*\*Affiliations\*\*[:：])\s*(.+)$", summary_text, flags=re.MULTILINE)
        if match:
            candidate = match.group(1).strip()
            if candidate and candidate.lower() not in {"not provided", "unknown"}:
                return candidate
        return fallback

    def _single_line(value: str) -> str:
        if not value:
            return ""
        return " ".join(value.split())

    ranked_list = list(ranked)
    subject = f"Daily LLM Safety Paper Digest ({date_descriptor})"
    lines: List[str] = [
        f"# Daily LLM Safety Paper Digest ({date_descriptor})",
        "",
        f"*Research Focus:* {focus or 'LLM Safety'}",
        f"*Data Sources:* {sources_descriptor or 'Unknown'}",
        "",
    ]
    if not ranked_list:
        lines.append("No papers found matching the criteria today. Consider lowering the threshold or expanding the search range.")
    else:
        for paper in iter_with_progress(
            ranked_list,
            description="Building report entries",
            total=len(ranked_list),
        ):
            summary_raw = (paper.summary or paper.paper.summary or "").strip()
            reasoning = (paper.reasoning or "").strip()
            if reasoning:
                reasoning = reasoning.replace("\n", " ").strip()
            categories = ", ".join(paper.paper.categories) if paper.paper.categories else "Not provided"
            title_text = " ".join(paper.paper.title.split())
            title_line = f"{paper.rank}. {title_text}"
            lines.extend(
                [
                    f"**{title_line}**",
                    f"- **Topic:** {paper.main_topic or 'Other'}",
                    f"- **LLM Assessment:** {reasoning or 'No explanation'}",
                    f"- **arXiv Categories:** {categories}",
                    f"- **Authors:** {_format_authors(paper.paper.authors)}",
                    f"- **Source:** {paper.paper.source}",
                    f"- **Link:** {paper.paper.link}",
                ]
            )
            institutions = _extract_institutions_from_summary_text(summary_raw, paper.paper.affiliations)
            lines.append(f"- **Institutions:** {institutions}")
            summary_text = summary_raw.strip()
            abstract_text = _single_line(paper.paper.summary or "")
            if include_llm_analysis and summary_text:
                lines.append("- **Summary (LLM Analysis):**")
                summary_lines = []
                for line in summary_text.splitlines():
                    stripped = line.strip()
                    # Skip empty lines and metadata lines we extract separately
                    if not stripped:
                        continue
                    if stripped.startswith("🏛️"):
                        continue
                    # Skip the "Affiliations" line in the summary body if we've already extracted it
                    if stripped.startswith("- **Affiliations**"):
                        continue
                    summary_lines.append(f"  {line}")
                if summary_lines:
                    lines.extend(summary_lines)
                else:
                    lines.extend([f"  {line}" for line in summary_text.splitlines() if line.strip()])
                lines.append(f"- **Abstract:** {abstract_text or 'Not provided'}")
            else:
                fallback_summary = abstract_text or _single_line(summary_text)
                lines.append(f"- **Summary:** {fallback_summary or 'No summary'}")
            lines.append("")
    lines.append("---")
    body = "\n".join(lines).strip()
    LOGGER.info("Generated Markdown report with %d entries.", len(ranked_list))
    return subject, body


def summarize_ranked_papers(
    llm: LLMClient,
    ranked: Sequence[RankedPaper],
    *,
    tldr_language: str = "English",
    max_attempts: int = 3,
) -> List[RankedPaper]:
    ranked_list = list(ranked)
    if not ranked_list:
        return []
    payload = SUMMARY_USER_PROMPT_TEMPLATE.format(
        language=tldr_language,
        papers=_format_ranked_papers_for_summary(ranked_list),
    )
    messages = [
        LLMMessage(role="system", content=SUMMARY_SYSTEM_PROMPT),
        LLMMessage(role="user", content=payload),
    ]
    last_response_content: Optional[str] = None

    def _with_fallback_summaries() -> List[RankedPaper]:
        LOGGER.warning(
            "Falling back to existing summaries for %d papers due to LLM summary failures.",
            len(ranked_list),
        )
        updated: List[RankedPaper] = []
        for paper in ranked_list:
            fallback = paper.summary or paper.paper.summary
            updated.append(paper.model_copy(update={"summary": fallback}))
        return updated

    def _parse_sectioned_summaries(raw_content: str) -> dict[str, str]:
        summary_map: dict[str, str] = {}
        if not raw_content:
            return summary_map
        blocks = re.split(r"\n\s*-{3,}\s*\n", raw_content.strip())
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            header = lines[0].strip()
            # Robust extraction for "Paper ID: <id>" pattern
            match = re.search(r"paper\s*id\s*[:：]\s*(.*)", header, re.IGNORECASE)
            if not match:
                LOGGER.debug("Skipping summary block without Paper ID header: %s", block)
                continue
            # Extract ID and clean up any remaining markdown chars (e.g. trailing **)
            paper_id = match.group(1).strip().strip("*")
            body = "\n".join(lines[1:]).strip()
            if paper_id and body:
                summary_map[paper_id] = body
            else:
                LOGGER.debug("Incomplete summary block for paper '%s': %s", paper_id, block)
        return summary_map

    summary_map: dict[str, str] = {}
    for attempt in range(1, max_attempts + 1):
        response = llm.chat_completion(messages)
        last_response_content = (response.content or "").strip()
        summary_map = _parse_sectioned_summaries(last_response_content)
        if summary_map:
            break
        LOGGER.warning(
            "Attempt %s/%s to generate summaries returned no valid entries. Raw response: %s",
            attempt,
            max_attempts,
            last_response_content,
        )
        if attempt == max_attempts:
            LOGGER.error(
                "Failed to obtain narrative summaries after %s attempts. Last response: %s",
                max_attempts,
                last_response_content,
            )
            return _with_fallback_summaries()

    updated: List[RankedPaper] = []
    for paper in iter_with_progress(
        ranked_list,
        description="Attaching narrative summaries",
        total=len(ranked_list),
    ):
        replacement = summary_map.get(paper.paper.id)
        if replacement:
            updated.append(paper.model_copy(update={"summary": replacement}))
        else:
            fallback = paper.summary or paper.paper.summary
            updated.append(paper.model_copy(update={"summary": fallback}))
            LOGGER.warning(
                "No LLM summary returned for paper %s; using existing summary.", paper.paper.id
            )
    LOGGER.info("Generated %d narrative summaries for ranked papers.", len(updated))
    return updated


def _apply_cached_summaries(
    cache: CacheManager | None,
    ranked: Sequence[RankedPaper],
) -> Tuple[List[RankedPaper], int]:
    ranked_list = list(ranked)
    if not ranked_list:
        return [], 0
    if cache is None:
        missing = sum(1 for paper in ranked_list if not (paper.summary or "").strip())
        return ranked_list, missing
    updated: List[RankedPaper] = []
    missing = 0
    for paper in iter_with_progress(
        ranked_list,
        description="Applying summaries",
        total=len(ranked_list),
    ):
        summary = (paper.summary or "").strip()
        if summary:
            updated.append(paper.model_copy(update={"summary": summary}))
            continue
        cached = cache.load_summary(paper.paper)
        if cached:
            updated.append(paper.model_copy(update={"summary": cached}))
        else:
            updated.append(paper)
            missing += 1
    return updated, missing


def _format_ranked_papers_for_summary(papers: Sequence[RankedPaper]) -> str:
    formatted: List[str] = []
    for paper in iter_with_progress(
        papers,
        description="Formatting ranked papers",
        total=len(papers),
    ):
        authors = ", ".join(paper.paper.authors[:10])
        affiliations = ", ".join(paper.paper.affiliations[:10]) or "Unknown"
        existing_summary = (paper.summary or "").strip()
        full_text = _extract_full_text(paper)
        lines = [
            f"- paper_id: {paper.paper.id}",
            f"  title: {paper.paper.title}",
            f"  authors: {authors}",
            f"  affiliations: {affiliations}",
            f"  link: {paper.paper.link}",
        ]
        if existing_summary:
            lines.append(f"  prior_summary: {existing_summary}")
        if full_text:
            # Use a delimited block for full text to help LLM parse it
            lines.append("  full_text: >>>")
            lines.append(full_text)
            lines.append("<<<")
        else:
            lines.append("  full_text: [No content available]")
        formatted.append("\n".join(lines))
    return "\n\n".join(formatted)


def _extract_full_text(paper: RankedPaper) -> str:
    full_text = getattr(paper.paper, "full_text", None)
    if isinstance(full_text, str):
        normalized = full_text.strip()
        if normalized:
            return normalized
    return (paper.paper.summary or "").strip()



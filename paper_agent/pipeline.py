from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
)
from .llm.utils import prepare_llm_json_content
from .mailer import EmailClient
from .models import ClassifiedPaper, PipelineResult, PipelineSettings, RankedPaper, RawPaper
from .rankers import HybridRanker
from .keywords import resolve_keywords

LOGGER = logging.getLogger(__name__)

L1_REQUIRED_CATEGORIES = {"cs.CL", "cs.LG", "cs.AI"}
L1_OPTIONAL_CATEGORIES = {"cs.CR", "cs.CY"}


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
    total_steps = 7 + (1 if should_send_email else 0)
    current_step = 0

    def log_step(description: str) -> int:
        nonlocal current_step
        current_step += 1
        LOGGER.info("[Step %d/%d] %s", current_step, total_steps, description)
        return current_step

    LOGGER.info(
        "Starting paper recommendation pipeline for %s (sources: %s, focus: %s, email=%s).",
        date_descriptor,
        source_descriptor,
        focus,
        "enabled" if should_send_email else "disabled",
    )
    cache = CacheManager()
    init_step_index = log_step("初始化数据源与客户端")
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
            "[Step %d/%d] 已准备 %d 个数据源：%s",
            init_step_index,
            total_steps,
            len(fetchers),
            ", ".join(fetcher.source_name for fetcher in fetchers),
        )
    else:
        LOGGER.info(
            "[Step %d/%d] 缓存命中，跳过数据源初始化（sources: %s）。",
            init_step_index,
            total_steps,
            source_descriptor,
        )

    raw_key: Optional[str] = None
    if raw_papers is None:
        fetch_step_index = log_step("拉取原始论文数据")
        fetch_kwargs = {
            "target_date": settings.target_date,
            "start_date": settings.start_date,
            "end_date": settings.end_date,
            "max_results": None,
        }
        aggregated: Dict[str, RawPaper] = {}
        total_collected = 0
        for idx, fetcher in enumerate(fetchers, start=1):
            LOGGER.info(
                "[Step %d/%d] 拉取进度 %d/%d -> 来源 %s",
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
            for paper in fetched:
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
        fetch_step_index = log_step("加载缓存的原始论文数据")
        LOGGER.info(
            "[Step %d/%d] 从缓存加载 %d 篇原始论文 (key=%s)。",
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
    filter_step_index = log_step("应用 Layer 1 过滤器")
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

    log_step("执行 LLM 分类")
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

    log_step("排序并选出核心论文")
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
    relevant_ranked, missing_summaries = _apply_cached_summaries(cache, relevant_ranked)
    summary_step_index = log_step("生成 LLM 摘要")
    if missing_summaries > 0:
        LOGGER.info(
            "[Step %d/%d] 将为 %d 篇论文生成或更新摘要，其余摘要来自缓存。",
            summary_step_index,
            total_steps,
            missing_summaries,
        )
        relevant_ranked = summarize_ranked_papers(llm, relevant_ranked)
        for ranked_paper in relevant_ranked:
            if cache and ranked_paper.summary:
                cache.store_summary(ranked_paper.paper, ranked_paper.summary)
    else:
        LOGGER.info(
            "[Step %d/%d] 全部摘要从缓存加载，跳过 LLM 调用。",
            summary_step_index,
            total_steps,
        )
    log_step("构建日报内容")
    email_subject, email_body = build_daily_report(
        relevant_ranked,
        focus,
        date_descriptor,
        source_descriptor,
    )

    if should_send_email:
        log_step("发送邮件通知")
        EmailClient(email_config).send_markdown_email(
            subject=email_subject,
            body_markdown=email_body,
            receiver=settings.receiver_email,
        )

    LOGGER.info(
        "Pipeline completed for %s. 最终报告共包含 %d 篇论文。",
        date_descriptor,
        len(relevant_ranked),
    )

    return PipelineResult(
        settings=settings,
        papers=relevant_ranked,
        email_subject=email_subject,
        email_body=email_body,
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
    for paper in candidates:
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
    if missing_summaries > 0:
        LOGGER.info(
            "Generating new summaries for %d gap-fill papers.",
            missing_summaries,
        )
        relevant_ranked = summarize_ranked_papers(llm, relevant_ranked)
        for ranked_paper in relevant_ranked:
            if ranked_paper.summary:
                cache.store_summary(ranked_paper.paper, ranked_paper.summary)
    date_descriptor = f"{week_start.isoformat()} 至 {week_end.isoformat()}（关键词补漏）"
    _, email_body = build_daily_report(
        relevant_ranked,
        focus,
        date_descriptor,
        source_descriptor,
    )
    email_subject = f"每周 LLM Safety 补漏 ({week_start.isoformat()} 至 {week_end.isoformat()})"
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
    )


def _instantiate_fetchers(source_names: Sequence[str]) -> List[PaperFetcher]:
    fetchers: List[PaperFetcher] = []
    for source in source_names:
        try:
            fetchers.append(get_fetcher(source))
        except ValueError as exc:
            LOGGER.error("Unable to initialize fetcher for source '%s': %s", source, exc)
    return fetchers


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
    required = {category.lower() for category in L1_REQUIRED_CATEGORIES}
    optional = {category.lower() for category in L1_OPTIONAL_CATEGORIES}
    paper_list = list(papers)
    retained: List[RawPaper] = []
    missing_categories = 0
    bypassed = 0
    for paper in paper_list:
        if paper.source != "arxiv":
            retained.append(paper)
            bypassed += 1
            continue
        categories = {category.lower() for category in paper.categories}
        if not categories:
            missing_categories += 1
            continue
        has_required = bool(categories & required)
        has_optional = bool(categories & optional)
        if has_required or has_optional:
            retained.append(paper)
    LOGGER.info(
        "Category filter retained %d papers out of %d (skipped %d without categories, bypassed %d non-arXiv sources).",
        len(retained),
        len(paper_list),
        missing_categories,
        bypassed,
    )
    return retained


def filter_papers_by_keywords(papers: Iterable[RawPaper], keywords: Iterable[str]) -> List[RawPaper]:
    keyword_list = [keyword.strip().lower() for keyword in keywords if str(keyword).strip()]
    LOGGER.info("Applying keyword filter with %d keywords.", len(keyword_list))
    paper_list = list(papers)
    if not keyword_list:
        LOGGER.debug("No keywords provided; returning all %d papers.", len(paper_list))
        return paper_list
    filtered: List[RawPaper] = []
    for paper in paper_list:
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
    for paper in paper_list:
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
    keywords_text = ", ".join(keywords) if keywords else "未提供"
    required_keywords_text = ", ".join(required_keywords) if required_keywords else "无"

    for paper in paper_list:
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

        for paper, result in zip(pending_papers, batched_results):
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
    for paper in paper_list:
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
) -> tuple[str, str]:
    ranked_list = list(ranked)
    subject = f"每日 LLM Safety 论文速递 ({date_descriptor})"
    lines: List[str] = [
        f"# 每日 LLM Safety 论文速递 ({date_descriptor})",
        "",
        f"*聚焦领域:* {focus or 'LLM Safety'}",
        f"*数据源:* {sources_descriptor or '未知'}",
        "",
    ]
    if not ranked_list:
        lines.append("今日未找到满足条件的论文，建议调低阈值或扩大检索范围。")
    else:
        for paper in ranked_list:
            summary = (paper.summary or paper.paper.summary or "").strip()
            if len(summary) > 0:
                summary = summary.replace("\n", " ").strip()
            reasoning = (paper.reasoning or "").strip()
            if len(reasoning) > 0:
                reasoning = reasoning.replace("\n", " ").strip()
            categories = ", ".join(paper.paper.categories) if paper.paper.categories else "未提供"
            title_line = f"{paper.rank}. {paper.paper.title} (分数: {paper.relevance_score:.2f})"
            lines.extend(
                [
                    f"**{title_line}**",
                    f"- **主题:** {paper.main_topic or 'Other'}",
                    f"- **LLM 评估:** {reasoning or '暂无说明'}",
                    f"- **摘要:** {summary or '暂无摘要'}",
                    f"- **arXiv 分类:** {categories}",
                    f"- **来源:** {paper.paper.source}",
                    f"- **链接:** {paper.paper.link}",
                    "",
                ]
            )
    lines.append("---")
    body = "\n".join(lines).strip()
    LOGGER.info("Generated Markdown report with %d entries.", len(ranked_list))
    return subject, body


def summarize_ranked_papers(
    llm: LLMClient,
    ranked: Sequence[RankedPaper],
    *,
    max_attempts: int = 3,
) -> List[RankedPaper]:
    ranked_list = list(ranked)
    if not ranked_list:
        return []
    payload = SUMMARY_USER_PROMPT_TEMPLATE.format(
        papers=_format_ranked_papers_for_summary(ranked_list)
    )
    messages = [
        LLMMessage(role="system", content=SUMMARY_SYSTEM_PROMPT),
        LLMMessage(role="user", content=payload),
    ]
    last_error: Optional[Exception] = None
    last_response_content: Optional[str] = None
    data: Optional[dict] = None

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

    for attempt in range(1, max_attempts + 1):
        response = llm.chat_completion(messages)
        last_response_content = response.content
        try:
            parsed_content = prepare_llm_json_content(response.content, context="summaries")
            data = json.loads(parsed_content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            LOGGER.warning(
                "Attempt %s/%s to generate summaries failed: %s",
                attempt,
                max_attempts,
                exc,
                exc_info=True,
            )
            if attempt == max_attempts:
                LOGGER.error(
                    "Failed to obtain valid LLM summaries after %s attempts. Last response: %s",
                    max_attempts,
                    last_response_content,
                )
                return _with_fallback_summaries()
            continue
        else:
            break

    if data is None:
        LOGGER.error(
            "LLM summarization did not return data despite %s attempts. Last error: %s",
            max_attempts,
            last_error,
        )
        return _with_fallback_summaries()
    summary_map: dict[str, str] = {}
    for item in data.get("papers", []):
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if paper_id and summary:
            summary_map[paper_id] = summary
    updated: List[RankedPaper] = []
    for paper in ranked_list:
        replacement = summary_map.get(paper.paper.id)
        if replacement:
            updated.append(paper.model_copy(update={"summary": replacement}))
        else:
            fallback = paper.summary or paper.paper.summary
            updated.append(paper.model_copy(update={"summary": fallback}))
            LOGGER.warning(
                "No LLM summary returned for paper %s; using existing summary.", paper.paper.id
            )
    LOGGER.info("Generated %d abstractive summaries for ranked papers.", len(updated))
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
    for paper in ranked_list:
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
    for paper in papers:
        authors = ", ".join(paper.paper.authors[:10])
        existing_summary = (paper.summary or "").strip()
        full_text = _extract_full_text(paper)
        lines = [
            f"- paper_id: {paper.paper.id}",
            f"  title: {paper.paper.title}",
            f"  authors: {authors}",
            f"  link: {paper.paper.link}",
        ]
        if existing_summary:
            lines.append(f"  prior_summary: {existing_summary}")
        if full_text:
            lines.append(f"  full_text: {full_text}")
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



from __future__ import annotations

from typing import Sequence

from ..keywords import DEFAULT_LAYER1_KEYWORDS, DEFAULT_REQUIRED_KEYWORDS

CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior researcher who specialises in large language model safety (LLM Safety). You will evaluate one paper at a time.

First, read the title, abstract, and arXiv categories. Determine whether the paper is relevant to LLM Safety. LLM Safety includes (but is not limited to):
{keyword_section}{required_section}

Return **strict JSON** with the following schema:
{{
  "is_relevant": <true|false>,
  "relevance_score": <float between 0 and 1>,
  "reasoning": "<concise justification (<=60 words)>",
  "main_topic": "<Alignment | Bias | Privacy | Red Teaming | Adversarial Attacks | Defences | Robustness | Interpretability | Watermarking | Hallucination | Other>"
}}

Guidelines:
- Set is_relevant to true only when the relevance_score is >= 0.7.
- Provide a short reasoning referencing the key idea.
- Choose the closest main_topic; use "Other" if nothing fits.
- Respond with JSON only. No prose, no Markdown, no code fences.
"""


CLASSIFICATION_USER_PROMPT_TEMPLATE = """\
你是一位专门研究大型语言模型安全（LLM Safety）的资深研究员。请根据以下论文信息完成评估：

研究焦点: {focus}
关键词列表: {keywords}
必含关键词（至少命中其中一个）: {required_keywords}
论文题目: {title}
摘要: {abstract}
arXiv 分类: {categories}

请仅输出 JSON，且字段必须严格为 is_relevant、relevance_score、reasoning、main_topic。
"""


EMAIL_SYSTEM_PROMPT = """\
You are a scientific newsletter curator. Summarize papers into a crisp email digest.
Output Markdown only. Use short paragraphs and bullet lists. Always include subject and body.
Return JSON:
{
  "subject": "<string>",
  "body": "<markdown string>"
}
"""

EMAIL_USER_PROMPT_TEMPLATE = """\
Audience research focus:
{target}

Ranked papers:
{papers}
"""


TOPIC_EXPANSION_SYSTEM_PROMPT = """\
You are a research domain expert helping to expand search queries.
Given base research topics and their subtopics, produce concise synonym lists that improve keyword recall.

Return JSON strictly matching this schema:
{
  "topics": [
    {
      "basic_topic": "<string>",
      "synonyms": ["synonym-1", "synonym-2"],
      "subtopics": [
        {
          "name": "<string>",
          "synonyms": ["synonym-1", "synonym-2"]
        }
      ]
    }
  ]
}

Include the original topic names in the synonym lists. Provide no more than 8 synonyms per list. Use domain-specific terminology.
"""


TOPIC_EXPANSION_USER_PROMPT_TEMPLATE = """\
Expand the following research topics and their subtopics with relevant synonyms suitable for literature search.
Topics (JSON):
{topics}
"""


SUMMARY_SYSTEM_PROMPT = """\
You are an expert scientific writer generating accurate, concise paper summaries.
For each paper, craft a short abstract that captures the main contribution, method, and findings.

Return JSON strictly matching this schema:
{
  "papers": [
    {
      "paper_id": "<string>",
      "summary": "<80-120 word abstractive summary>"
    }
  ]
}

Summaries must be factual, avoid speculation, and reference the provided content only.
"""


SUMMARY_USER_PROMPT_TEMPLATE = """\
Summarize each paper using the provided full text. Follow these rules:
- Highlight motivation, approach, and key results.
- Keep each summary under 120 words.
- Do not invent details that are not present in the text.

Papers:
{papers}
"""


def build_classification_system_prompt(
    keywords: Sequence[str] | None = None,
    required_keywords: Sequence[str] | None = None,
) -> str:
    keyword_list = list(keywords) if keywords else list(DEFAULT_LAYER1_KEYWORDS)
    required_list = list(required_keywords) if required_keywords else list(DEFAULT_REQUIRED_KEYWORDS)
    if keyword_list:
        keyword_section = "\n".join(f"- {keyword}" for keyword in keyword_list) + "\n"
    else:
        keyword_section = "- Focus on overall LLM Safety relevance (no specific keywords provided).\n"
    if required_list:
        required_section = (
            "Additionally, the paper must mention at least one of the following keywords:\n"
            + "\n".join(f"- {keyword}" for keyword in required_list)
            + "\n"
        )
    else:
        required_section = ""
    return CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE.format(
        keyword_section=keyword_section,
        required_section=required_section,
    )


CLASSIFICATION_SYSTEM_PROMPT = build_classification_system_prompt()


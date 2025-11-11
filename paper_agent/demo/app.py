from __future__ import annotations

import streamlit as st

from ..models import PipelineSettings
from ..pipeline import generate_recommendations


st.set_page_config(page_title="Paper Agent Demo", layout="wide")
st.title("📄 Paper Agent Demo")
st.write("输入研究方向，自动抓取论文并生成推荐邮件。")

with st.sidebar:
    st.header("配置")
    topics_input = st.text_area("研究焦点（每行一个，可选）", "LLM Safety")
    max_results = st.slider("每个方向抓取数量", min_value=1, max_value=20, value=6)
    send_email = st.checkbox("完成后发送邮件", value=False)
    receiver = st.text_input("收件人邮箱（可选，若留空使用默认配置）")
    run_button = st.button("运行推荐")

if run_button:
    topics = [topic.strip() for topic in topics_input.splitlines() if topic.strip()]
    if not topics:
        topics = ["LLM Safety"]
    with st.spinner("正在抓取、解析、调用 LLM..."):
        try:
            result = generate_recommendations(
                PipelineSettings(
                    topics=topics,
                    max_results_per_topic=max_results,
                    send_email=send_email,
                    receiver_email=receiver or None,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"运行失败：{exc}")
            st.stop()

    st.success("处理完成！")
    st.subheader("推荐邮件主题")
    st.write(result.email_subject)

    st.subheader("推荐邮件正文（Markdown）")
    st.markdown(result.email_body)

    st.subheader("排序结果")
    for paper in result.papers:
        with st.expander(f"{paper.rank}. {paper.paper.title}"):
            st.markdown(
                f"- 链接: [{paper.paper.link}]({paper.paper.link})\n"
                f"- 作者: {', '.join(paper.paper.authors)}\n"
                f"- 发布时间: {paper.paper.published.strftime('%Y-%m-%d')}\n"
                f"- arXiv 分类: {', '.join(paper.paper.categories) if paper.paper.categories else '未提供'}\n"
                f"- LLM 主题判断: {paper.main_topic or 'Other'}\n"
                f"- 相关性得分: {paper.relevance_score:.2f}\n"
                f"- 排序得分: {paper.score:.2f}\n"
                f"- LLM 解释: {paper.reasoning or '无'}"
            )


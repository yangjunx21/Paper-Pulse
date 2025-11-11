# Paper-Pulse
# paper_agent

最小可运行的基于 LLM 的论文分类与推荐系统，支持批量抓取 → 解析 → 分类 → 排序 → 邮件推送，并提供在线 Demo 页面。

## 功能概览

- **自动抓取**：支持从 arXiv、Hugging Face Daily Papers、NeurIPS 2025 论文集等多源获取内容，可按指定日期或时间段汇总。
- **内容解析**：抽取标题、作者、摘要、发布时间、分类标签等结构化信息。
- **关键词初筛**：基于论文标题 / 摘要与用户输入的关键词进行快速筛选。
- **LLM 分类**：调用大语言模型生成精炼摘要，并给出相关性评分与分类建议。
- **混合排序**：结合 LLM 评分与发布时间对候选论文排序。
- **邮件推送**：按邮件格式生成推荐摘要，并通过 SMTP 发送。
- **交互 Demo**：提供 Streamlit 页面，支持在线体验与一键发送邮件。

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

| 环境变量            | 说明                              |
| ------------------- | --------------------------------- |
| `OPENAI_API_KEY`    | OpenAI (或兼容) 大模型 API Key    |
| `OPENAI_MODEL`      | （可选）模型名称，默认 `gpt-4o`   |
| `OPENAI_BASE_URL`   | （可选）API Base URL，默认 `http://115.182.62.174:18888/v1` |
| `EMAIL_HOST`        | SMTP 服务器地址                   |
| `EMAIL_PORT`        | SMTP 端口（默认 587）             |
| `EMAIL_USERNAME`    | SMTP 用户名                       |
| `EMAIL_PASSWORD`    | SMTP 密码或应用专用密码           |
| `EMAIL_SENDER`      | 发件人邮箱                        |
| `EMAIL_SENDER_NAME` | （可选）收件人看到的显示名称      |
| `EMAIL_RECEIVER`    | 默认收件人邮箱，可在界面覆盖      |

可以在项目根目录创建一个 `.env` 文件集中维护这些变量（默认会自动加载）：

```
OPENAI_API_KEY=sk-xxxx
OPENAI_MODEL=gpt-4o
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_USERNAME=bot@example.com
EMAIL_PASSWORD=app-specific-password
EMAIL_SENDER=bot@example.com
EMAIL_SENDER_NAME=Paper Pulse
EMAIL_RECEIVER=user@example.com
```

如果你的 `.env` 文件不在项目根目录，可以通过设置 `PAPER_AGENT_ENV_FILE` 环境变量来指定完整路径。

### 3. 命令行运行

```bash
python -m paper_agent.cli --topics "large language models" "graph neural networks" --date 2025-11-10 --sources arxiv huggingface_daily --max-results 12 --send-email --log-level INFO
```

也可以一次性抓取某个时间段内的投稿：

```bash
python -m paper_agent.cli --topics "mechanistic interpretability" --date-range 2025-11-08 2025-11-10 --max-results 16 --log-level DEBUG
```

**可选数据源**

- `arxiv`（默认）
- `huggingface_daily`
- `neurips_2025`

通过 `--sources` 参数可一次选择多个数据源，系统会自动去重并标注来源。

### 关键词配置

默认情况下，系统会使用内置的一组 LLM Safety 关键词用于 Layer 1 过滤和 LLM 提示词。你可以通过提供一个 YAML 配置文件完全自定义这份关键词表，以保持筛选逻辑与 Prompt 描述一致。

示例 `keywords.yaml`：

```
keywords:
  - safety
  - alignment
  - red teaming
  - jailbreak
  - mechanistic interpretability
```

也支持分组写法（会自动展开成扁平列表）：

```
keyword_groups:
  - name: 防御
    keywords:
      - adversarial defense
      - poisoning
  - name: 可靠性
    keywords:
      - robustness
      - trustworthy
```

如果希望“必含”某些词（例如必须包含 LLM 相关术语），可在同一个 YAML 中添加 `required_keywords` 或 `required_keyword_groups`：

```
keywords:
  - safety
required_keywords:
  - LLM
  - "Large Language Model"
required_keyword_groups:
  - keywords:
      - GPT
      - Transformer
```

系统会在 Layer 1 过滤阶段强制至少命中其中一个必含词，并在 LLM Prompt 中同步展示。

也可通过 CLI 参数直接传入必含词：

```bash
python -m paper_agent.cli --required-keywords LLM "Large Language Model" ...
```

运行时通过 `--keywords-file` 指定：

```bash
python -m paper_agent.cli --keywords-file /abs/path/to/keywords.yaml ...
```

或者设置环境变量 `PAPER_AGENT_KEYWORDS_FILE=/abs/path/to/keywords.yaml`，CLI 和 Demo 页面都会自动加载。

### 4. 启动 Demo 页面

```bash
streamlit run paper_agent/demo/app.py
```

## 架构说明

```
paper_agent/
├── paper_agent/
│   ├── config.py           # 全局配置 & 环境变量加载
│   ├── models.py           # 数据类定义
│   ├── pipeline.py         # 主流程调度
│   ├── cli.py              # 命令行入口
│   ├── llm/
│   │   ├── client.py       # LLM 客户端封装
│   │   └── prompts.py      # Prompt 模板
│   ├── fetchers/
│   │   ├── arxiv_fetcher.py
│   │   ├── hf_daily_fetcher.py
│   │   ├── neurips_fetcher.py
│   │   └── base.py
│   ├── parsers/
│   │   └── arxiv_parser.py
│   ├── rankers/
│   │   └── hybrid_ranker.py
│   └── mailer/
│       └── email_client.py
└── requirements.txt
```

模块之间通过定义良好的接口解耦，方便后续替换数据源、模型或排序策略。

## 扩展建议

- **数据源**：补充 Semantic Scholar、Twitter Lists 等抓取器。
- **多模型融合**：加入传统关键词匹配或 embedding 检索，与 LLM 结果融合。
- **反馈闭环**：记录用户点击 / 邮件打开情况，进一步优化排序。
- **多渠道推送**：扩展到 Slack、企业微信等渠道。
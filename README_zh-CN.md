# Paper Pulse 🚀

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/lang-中文-red.svg)](README_zh-CN.md)

![Paper Pulse Abstract](figs/abstract.jpg)

**Paper Pulse** 是一个极简但功能强大的基于 LLM 的学术论文发现、分类和总结系统。它自动化了从各种来源（ArXiv、Hugging Face 等）获取论文、基于用户意图进行过滤、使用 LLM 进行分析并通过电子邮件发送结构化报告的流程。

## ✨ 核心功能

- **多源获取**：目前支持 **ArXiv**、**Hugging Face Daily Papers** 和 **NeurIPS 2025**。我们正在持续完善对所有主流 ML 会议和其他信息源的支持。
- **意图解析代理**：将自然语言描述（例如 *“我对 LLM 的越狱攻击感兴趣”*）转换为具有优化关键词的结构化搜索配置文件。
- **智能过滤**：
  - **第 1 层（关键词）**：使用 Trie/Set 匹配进行快速预过滤。
  - **第 2 层（LLM）**：由 LLM 进行深度语义相关性评分和推理。
- **混合排名**：根据 LLM 相关性得分和新鲜度对论文进行排序。
- **深度分析**：下载 PDF 以提取全文并生成结构化摘要（背景、创新点、方法、实验）。
- **邮件投递**：发送格式精美的 Markdown 报告直接到您的收件箱。

![Paper Pulse Framework](figs/framework.jpg)

## 🚀 快速开始

### 先决条件

- Python 3.9+
- OpenAI API Key (或兼容的 LLM 端点)

### 安装

1. **克隆仓库：**
   ```bash
   git clone https://github.com/yourusername/paper-pulse.git
   cd paper-pulse
   ```

2. **设置虚拟环境：**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 用户使用: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **配置环境：**
   在根目录创建一个 `.env` 文件：
   ```env
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-4o
   # OPENAI_BASE_URL=... (可选)

   # 邮件设置 (邮件投递需要)
   EMAIL_HOST=smtp.gmail.com / smtp.163.com
   EMAIL_PORT=587 / 465
   EMAIL_USERNAME=your-email@gmail.com
   EMAIL_PASSWORD=your-app-password
   EMAIL_SENDER=your-email@gmail.com
   EMAIL_RECEIVER=target-email@example.com
   ```

   > **注意：** 如果您不需要邮件通知，可以跳过 `EMAIL_*` 配置。生成的报告将保存在本地的 `reports/` 目录中。

## 📖 使用方法

### 1. 基于意图的模式（推荐）

让“意图代理”帮助您构建搜索配置文件。

**步骤 1：构建配置文件**
运行交互式构建器来定义您的研究兴趣。
```bash
./scripts/build_intent_profile.sh "my_research_focus"
# 按照提示描述您正在寻找的内容。
```

**步骤 2：运行主程序**
使用您刚刚创建的配置文件执行主程序。
```bash
# 将您的配置文件名称设置为环境变量
export PROFILE_NAME="default"
./scripts/run_with_intent.sh
```
*您可以在 `scripts/run_with_intent.sh` 中或通过环境变量（例如 `DATE_RANGE_START`）自定义参数。*

### 2. CLI 模式（手动）

您也可以直接运行 CLI 进行一次性搜索。

```bash
python -m paper_agent.cli \
  --topics "mechanistic interpretability" "sparse autoencoders" \
  --date 2025-11-20 \
  --sources arxiv huggingface_daily \
  --max-results 10 \
  --send-email
```

## 📂 项目结构

```
paper-pulse/
├── config/              # 配置和配置文件
│   └── intent_profiles/ # 由意图代理生成的 JSON 配置文件
├── paper_agent/         # 核心包
│   ├── llm/             # 提示词和 LLM 客户端包装器
│   ├── fetchers/        # 来源适配器 (ArXiv, HF, etc.)
│   ├── parsers/         # PDF 和文本处理
│   ├── pipeline.py      # 主要处理逻辑
│   └── intent_agent.py  # 配置文件生成逻辑
├── scripts/             # 辅助脚本
│   ├── build_intent_profile.sh
│   └── run_with_intent.sh
└── reports/             # 生成的 Markdown 报告 (本地副本)
```

## 🛠 配置

您可以通过 CLI 参数或 `.env` 文件调整管道行为。关键环境变量：

| 变量名 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | 您的 LLM API 密钥。 | 必填 |
| `PAPER_PULSE_LANG` | 总结使用的语言 (例如 "Chinese", "English")。 | English |
| `ENABLE_PDF_ANALYSIS` | 设置为 `true` 以启用 PDF 下载、全文提取和深度总结。 | `false` |
| `RELEVANCE_THRESHOLD` | 纳入报告的最低 LLM 相关性评分 (0.0-1.0)。 | `0.8` |
| `EMAIL_*` | 用于报告投递的 SMTP 设置。 | 可选 |

> **💡 提示：** 启用 `ENABLE_PDF_ANALYSIS=true` 可以获得更丰富的见解（方法论、实验等），但这会消耗更多的 Token 和时间。

## 🖊️ 引用

如果您觉得本项目有用，请引用：

```bibtex
@misc{yang2025paperpulse,
  title  = {Paper Pulse: An LLM-Based Academic Paper Discovery and Analysis System},
  author = {Junxiao Yang},
  year   = {2025},
  url    = {https://github.com/yangjunx21/Paper-Pulse}
}
```

## 📄 许可证

[MIT License](LICENSE)


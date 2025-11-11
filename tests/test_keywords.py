from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from paper_agent.paper_agent.keywords import (
    DEFAULT_LAYER1_KEYWORDS,
    KeywordConfig,
    resolve_keywords,
)


class ResolveKeywordsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure environment variable does not interfere with tests.
        self._original_env = os.environ.copy()
        os.environ.pop("PAPER_AGENT_KEYWORDS_FILE", None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_explicit_keywords_take_precedence(self) -> None:
        config = resolve_keywords(["Safety", "alignment", "safety "])
        self.assertIsInstance(config, KeywordConfig)
        self.assertEqual(config.keywords, ["Safety", "alignment"])
        self.assertEqual(config.required_keywords, [])

    def test_load_keywords_from_yaml_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write("keywords:\n  - alpha\n  - beta\n")
            path = tmp.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        config = resolve_keywords(keyword_file=path)
        self.assertEqual(config.keywords, ["alpha", "beta"])
        self.assertEqual(config.required_keywords, [])

    def test_load_keywords_from_grouped_yaml(self) -> None:
        content = textwrap.dedent(
            """
            keyword_groups:
              - name: 安全
                keywords:
                  - defence
                  - Security
              - name: 可靠性
                items:
                  - robustness
                  - trustworthy
              - misc:
                  terms:
                    - hallucination
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write(content)
            path = tmp.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        config = resolve_keywords(keyword_file=path)
        self.assertEqual(
            config.keywords,
            ["defence", "Security", "robustness", "trustworthy", "hallucination"],
        )
        self.assertEqual(config.required_keywords, [])

    def test_load_required_keywords_from_yaml(self) -> None:
        content = textwrap.dedent(
            """
            keywords:
              - safety
            required_keywords:
              - LLM
              - "Large Language Model"
            required_keyword_groups:
              - keywords:
                  - GPT
                  - Transformer
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write(content)
            path = tmp.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        config = resolve_keywords(keyword_file=path)
        self.assertEqual(config.keywords, ["safety"])
        self.assertEqual(
            config.required_keywords,
            ["LLM", "Large Language Model", "GPT", "Transformer"],
        )

    def test_default_keywords_used_when_no_input(self) -> None:
        config = resolve_keywords()
        self.assertEqual(list(DEFAULT_LAYER1_KEYWORDS), config.keywords)
        self.assertEqual([], config.required_keywords)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


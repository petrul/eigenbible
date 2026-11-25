"""Summarization strategies: ask an Ollama chat model to name the shared theme
of a handful of related passages (Bible chapters, historical chronicles, etc.).

Both strategies share the same request/response machinery (`Summarizer.label`);
only prompt construction differs (`ChapterSummarizer` vs `ShuffledLinesSummarizer`).
"""
import logging
import random
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)


class Summarizer(ABC):
    """Base for turning a group of related text passages into one concise label via Ollama."""

    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.model = model
        self._cache: dict[str, str] = {}
        logger.info(
            "Labelling with Ollama model '%s' at %s (%s)", model, self.url, type(self).__name__
        )

    @abstractmethod
    def _build_prompt(self, texts: list[str]) -> str:
        raise NotImplementedError

    def label(self, texts: list[str]) -> str:
        prompt = self._build_prompt(texts)
        cached = self._cache.get(prompt)
        if cached is not None:
            return cached
        response = requests.post(
            self.url,
            # think=False: qwen3.5 is a hybrid reasoning model that otherwise burns its
            # generation budget on hidden thinking output for long prompts.
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": "6h",
                "options": {"num_predict": 44, "temperature": 0.2},
            },
            timeout=300,
        )
        response.raise_for_status()
        label = response.json()["response"].strip()
        if not label:
            logger.warning("Ollama returned an empty label - keeping it empty rather than failing the run")
        self._cache[prompt] = label
        return label


class ChapterSummarizer(Summarizer):
    """Feeds each passage's opening excerpt to the model intact and separated."""

    SNIPPET_CHARS = 600  # per-neighbour text budget for small K
    TOTAL_PROMPT_CHARS = 3600  # fast GPU prefill & crisp focus even when K=30
    MAX_PROMPT_TEXTS = 12  # label from the 12 strongest passages while storing all K=30 refs

    def __init__(self, base_url: str, model: str, source_kind: str = "passages"):
        super().__init__(base_url, model)
        self.source_kind = source_kind

    def _build_prompt(self, texts: list[str]) -> str:
        selected = texts[: self.MAX_PROMPT_TEXTS]
        n = max(1, len(selected))
        per_text = self.SNIPPET_CHARS if n <= 6 else max(220, min(self.SNIPPET_CHARS, self.TOTAL_PROMPT_CHARS // n))
        passages = "\n\n---\n\n".join(text[:per_text] for text in selected)
        return (
            f"The following are excerpts from {len(selected)} {self.source_kind} that were found to be "
            "closely related in embedding space. In a few words, or at most one short sentence, "
            "name the theme or idea they share (answer in English). Answer with only the label "
            "itself - no preamble, no quotes.\n\n"
            f"{passages}\n\nShared theme:"
        )


ExcerptSummarizer = ChapterSummarizer


class ShuffledLinesSummarizer(Summarizer):
    """Pools lines from every retrieved passage together and shuffles them before
    prompting the LLM so the label relies on line-level theme/vocabulary rather
    than chapter boundaries or narrative order.
    """

    MAX_LINES = 60

    def _build_prompt(self, texts: list[str]) -> str:
        lines = [line.strip() for text in texts for line in text.splitlines() if line.strip()]
        random.shuffle(lines)
        shuffled = "\n".join(lines[: self.MAX_LINES])
        return (
            "Here are a few random sequences. Summarize their gist using a few words, "
            "maximum a sentence. Answer with only the label itself - no preamble, no "
            "quotes.\n\n"
            f"{shuffled}\n\nLabel:"
        )

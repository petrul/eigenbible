"""Summarization strategies: ask an Ollama chat model to name the shared theme

of a handful of bible passages. Both strategies below share the same
request/response machinery (Summarizer.label); only how the prompt is built
differs - see ChapterSummarizer vs ShuffledLinesSummarizer.
"""
import logging
import random
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)


class Summarizer(ABC):
    """Base for anything that turns a handful of chapter texts (as many as the

    caller's -k/--neighbours picked per component - see NeighbourhoodStrategy)
    into one label via an Ollama chat model. Subclasses only need to build the
    prompt (_build_prompt); the request/response plumbing is shared here.
    """

    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.model = model
        logger.info(
            "Labelling with Ollama model '%s' at %s (%s)", model, self.url, type(self).__name__
        )

    @abstractmethod
    def _build_prompt(self, texts: list[str]) -> str:
        raise NotImplementedError

    def label(self, texts: list[str]) -> str:
        prompt = self._build_prompt(texts)
        response = requests.post(
            self.url,
            # think=False: qwen3.5:4b is a hybrid reasoning model that otherwise burns its
            # whole generation budget on a hidden "thinking" field for prompts this long,
            # frequently leaving the actual "response" empty (and taking ~80s to do it).
            json={"model": self.model, "prompt": prompt, "stream": False, "think": False},
            timeout=300,
        )
        response.raise_for_status()
        label = response.json()["response"].strip()
        if not label:
            logger.warning("Ollama returned an empty label for a component - keeping it empty rather than failing the run")
        return label


class ChapterSummarizer(Summarizer):
    """Feeds each neighbour chapter's own opening excerpt to the model intact

    and kept separate from the others - the model sees each passage as a
    coherent whole, in its own voice."""

    SNIPPET_CHARS = 600  # per-neighbour text budget kept in the labelling prompt

    def _build_prompt(self, texts: list[str]) -> str:
        passages = "\n\n---\n\n".join(text[:self.SNIPPET_CHARS] for text in texts)
        return (
            "The following are excerpts from several bible chapters that were found to be "
            "closely related. In a few words, or at most one short sentence, name the theme "
            "they share. Answer with only the label itself - no preamble, no quotes.\n\n"
            f"{passages}\n\nShared theme:"
        )


class ShuffledLinesSummarizer(Summarizer):
    """Alternative to ChapterSummarizer: instead of keeping each neighbour

    chapter's excerpt intact and separate, splits every retrieved chapter
    into individual lines, pools every line from every chapter together, and
    shuffles the whole pool before handing it to the model - so the label
    has to come from whatever theme/vocabulary survives at the line level,
    rather than from chapter boundaries or narrative order a model might
    otherwise latch onto.
    """

    MAX_LINES = 60  # keeps the prompt bounded regardless of how long/many the source chapters are

    def _build_prompt(self, texts: list[str]) -> str:
        lines = [line.strip() for text in texts for line in text.splitlines() if line.strip()]
        random.shuffle(lines)
        shuffled = "\n".join(lines[:self.MAX_LINES])
        return (
            "Here are a few random sequences. Summarize their gist using a few words, "
            "maximum a sentence. Answer with only the label itself - no preamble, no "
            "quotes.\n\n"
            f"{shuffled}\n\nLabel:"
        )

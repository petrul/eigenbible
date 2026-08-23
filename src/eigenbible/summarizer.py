import logging

import requests

logger = logging.getLogger(__name__)


class ChapterSummarizer:
    """Asks an Ollama chat model to name the shared theme of a handful of bible passages."""

    SNIPPET_CHARS = 600  # per-neighbour text budget kept in the labelling prompt

    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.model = model
        logger.info("Labelling with Ollama model '%s' at %s", model, self.url)

    def label(self, texts: list[str]) -> str:
        passages = "\n\n---\n\n".join(text[:self.SNIPPET_CHARS] for text in texts)
        prompt = (
            "The following are excerpts from several bible chapters that were found to be "
            "closely related. In a few words, or at most one short sentence, name the theme "
            "they share. Answer with only the label itself - no preamble, no quotes.\n\n"
            f"{passages}\n\nShared theme:"
        )
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

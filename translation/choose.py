"""
translation/choose.py
Picking one English option for a word. The translator hands a Chooser the
word, the role it plays, its options from the lexicon and the sentence, and
takes the first of the ranking it gets back.

FirstChoice keeps the lexicon's order: wasona's gloss first, then how often
speakers translate the word that way. A model-backed chooser (a local model
through Ollama, or a hosted service) implements the same method and can rank
by context instead: "musi" is "funny" for a video but "have fun" for a person.
"""
from __future__ import annotations

from typing import Protocol


class Chooser(Protocol):
    def rank(self, word: str, role: str, options: list[str], sentence: str) -> list[tuple[str, float]]:
        """options, best first, each with a probability; the probabilities sum to 1."""
        ...


class FirstChoice:
    """The lexicon's own order, with probabilities falling off as 1/rank."""

    def rank(self, word: str, role: str, options: list[str], sentence: str) -> list[tuple[str, float]]:
        weights = [1 / (i + 1) for i in range(len(options))]
        total = sum(weights) or 1.0
        return [(option, w / total) for option, w in zip(options, weights)]

"""
translation/evaluate.py
Scores Toki Pona -> English against wasona's own translations.

    .venv/bin/python -m translation.evaluate            # summary and the 25 worst
    .venv/bin/python -m translation.evaluate --worst 60

Every example, exercise, accepted alternative and text line in
wasona_sentences.tsv is translated and compared with wasona's English using
chrF (Popović 2015): the F-score of shared character n-grams (n = 1 to 6,
recall weighted twice as much as precision), on lowercased text without
punctuation or spaces. It rewards near misses ("The bird eats bugs" against
"The bird eats this bug"), which suits loose, natural reference translations
where an exact match is rare. The score is the mean over sentences, 0-100.

wasona's English often adds what Toki Pona leaves to context (tense, number,
"the" or "a"), so 100 is out of reach; the score is for comparing versions.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from translation.to_english import EnglishRenderer, translate

_TSV = Path(__file__).parent.parent / "wasona_sentences.tsv"
KINDS = ("exercise", "also_accept", "example", "text")


def _normalise(text: str) -> str:
    """Lowercase, drop wasona's notes ("(lit. ...)"), punctuation and spaces."""
    text = re.sub(r"\([^)]*\)", "", text.lower())
    return re.sub(r"[^a-z0-9']", "", text)


def chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score of hypothesis against reference, 0-100."""
    hyp, ref = _normalise(hypothesis), _normalise(reference)
    precisions, recalls = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    return 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


@dataclass
class Scored:
    kind: str
    toki_pona: str
    reference: str
    english: str
    parsed: bool
    score: float


def pairs() -> list[tuple[str, str, str]]:
    """(kind, toki_pona, english) for every sentence wasona translates."""
    with open(_TSV, encoding="utf-8", newline="") as fh:
        return [(r["kind"], r["toki_pona"], r["english"]) for r in csv.DictReader(fh, delimiter="\t")
                if r["kind"] in KINDS and r["english"].strip()]


def evaluate() -> list[Scored]:
    renderer = EnglishRenderer()
    out = []
    for kind, tp, reference in pairs():
        sentences = translate(tp, renderer)
        english_ = " ".join(t.english for t in sentences)
        parsed = all(t.parsed for t in sentences)
        out.append(Scored(kind, tp, reference, english_, parsed, chrf(english_, reference)))
    return out


def mean(scores: list[Scored]) -> float:
    return sum(s.score for s in scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worst", type=int, default=25, help="how many of the worst to list")
    args = parser.parse_args()
    scored = evaluate()
    by_kind = defaultdict(list)
    for s in scored:
        by_kind[s.kind].append(s)
    print(f"chrF {mean(scored):.1f} over {len(scored)} sentences "
          f"({sum(not s.parsed for s in scored)} glossed word by word)")
    for kind in KINDS:
        print(f"  {kind:12} {mean(by_kind[kind]):5.1f}  ({len(by_kind[kind])})")
    print(f"\nWorst {args.worst}:")
    for s in sorted(scored, key=lambda s: s.score)[:args.worst]:
        print(f"  {s.score:5.1f}  {s.toki_pona}\n         ours:   {s.english}\n         wasona: {s.reference}")


if __name__ == "__main__":
    main()

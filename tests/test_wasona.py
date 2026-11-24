"""
tests/test_wasona.py
Runs every Toki Pona sentence from the wasona course through the validator.
wasona is the project's grammar authority (see README.md), so ideally every
sentence here is valid. Sentences come from wasona_sentences.tsv, which
build_wasona_corpus.py extracts from the compiled_sources/wasona submodule.

One-word vocabulary drills (kind "vocab") are left out: they are words, not
sentences. As of 2026-10-16 every remaining sentence parses and is valid.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from validator import validate_text

_TSV = Path(__file__).parent.parent / "wasona_sentences.tsv"
_SENTENCE_KINDS = {"example", "exercise", "also_accept", "text"}

# Pass counts recorded on 2026-10-16, out of 561 distinct sentences. Raise
# them as the grammar improves; a drop below them is a regression.
_PARSE_BASELINE = 561
_VALID_BASELINE = 561


def _sentences() -> list[str]:
    with open(_TSV, encoding="utf-8", newline="") as fh:
        rows = csv.DictReader(fh, delimiter="\t")
        return list(dict.fromkeys(r["toki_pona"] for r in rows
                                  if r["kind"] in _SENTENCE_KINDS))


def _failures(check) -> tuple[int, list[str]]:
    sentences = _sentences()
    failed = []
    for s in sentences:
        results = validate_text(s)
        if not (results and all(check(r) for r in results)):
            failed.append(s)
    return len(sentences) - len(failed), failed


@pytest.mark.slow
def test_wasona_parse_rate():
    passed, failed = _failures(lambda r: r.parse_count > 0)
    assert passed >= _PARSE_BASELINE, (
        f"wasona: only {passed} sentences parsed "
        f"(regression: expected >= {_PARSE_BASELINE}).\n"
        + "\n".join(f"  {s!r}" for s in failed[:20])
    )


@pytest.mark.slow
def test_wasona_valid_rate():
    passed, failed = _failures(lambda r: r.is_valid)
    assert passed >= _VALID_BASELINE, (
        f"wasona: only {passed} sentences valid "
        f"(regression: expected >= {_VALID_BASELINE}).\n"
        + "\n".join(f"  {s!r}" for s in failed[:20])
    )

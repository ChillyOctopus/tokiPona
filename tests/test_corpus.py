"""
tests/test_corpus.py
Corpus tests. Two corpora come from the Toki_Pona-Parser-master project and a
third holds what wasona discourages; all three are curated to agree with
wasona (each file's header lists the edits):

  private/tests-data/known_good.txt         — lines that should be valid
  private/tests-data/known_bad.txt          — lines that should be rejected
                                               (some sentence on the line is not valid)
  private/tests-data/known_discouraged.txt  — lines that are valid but get a WARNING

The corpora are other people's sentences, so they live in private/, which git
ignores; without them these tests are skipped.

Each line is split into sentences with validator.validate_text. Aggregate
baselines guard against regressions; spot checks cover sentences that must
always work. As of 2026-11-01 every line in all three files passes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from validator import ValidationResult, validate, validate_text

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_DATA = Path(__file__).parent.parent / "private" / "tests-data"
if not (_DATA / "known_good.txt").exists():
    pytest.skip("the corpora in private/tests-data aren't here", allow_module_level=True)
_RIGHT_PATH = _DATA / "known_good.txt"
_WRONG_PATH = _DATA / "known_bad.txt"
_DISCOURAGED_PATH = _DATA / "known_discouraged.txt"


def _load(path: Path) -> list[str]:
    """The corpus lines, skipping blank lines and # comments."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


@lru_cache(maxsize=None)
def _results(line: str) -> tuple[ValidationResult, ...]:
    return tuple(validate_text(line))


def _parses(line: str) -> bool:
    """Every sentence on the line has a parse."""
    results = _results(line)
    return bool(results) and all(r.parse_count for r in results)


def _is_valid(line: str) -> bool:
    """Every sentence on the line parses with no ERROR diagnostic."""
    results = _results(line)
    return bool(results) and all(r.is_valid for r in results)


def _is_warned(line: str) -> bool:
    """The line is valid, and some sentence on it gets a WARNING."""
    return _is_valid(line) and any(r.warnings for r in _results(line))


# ---------------------------------------------------------------------------
# Aggregate corpus tests (regression guards)
# ---------------------------------------------------------------------------

# Pass counts recorded on 2026-11-01. Raise them as the grammar improves;
# a drop below them is a regression.
_RIGHT_PARSE_BASELINE = 2141   # of 2141 right lines, every sentence parses
_RIGHT_VALID_BASELINE = 2141   # of 2141 right lines, every sentence is valid
_WRONG_BASELINE       = 141    # of 141 wrong lines, rejected
_DISCOURAGED_BASELINE = 17     # of 17 discouraged lines, valid with a WARNING


def _first(lines: list[str]) -> str:
    return "\n".join(f"  {s!r}" for s in lines[:20])


@pytest.mark.slow
def test_right_corpus_parse_rate():
    """At least _RIGHT_PARSE_BASELINE right lines must parse."""
    sentences = _load(_RIGHT_PATH)
    failures = [s for s in sentences if not _parses(s)]
    passed = len(sentences) - len(failures)
    assert passed >= _RIGHT_PARSE_BASELINE, (
        f"Right corpus: only {passed}/{len(sentences)} parsed "
        f"(regression: expected >= {_RIGHT_PARSE_BASELINE}).\n"
        f"Failures (first 20):\n" + _first(failures)
    )


@pytest.mark.slow
def test_right_corpus_valid_rate():
    """
    At least _RIGHT_VALID_BASELINE right lines must be valid. This catches
    validator rules that reject good Toki Pona.
    """
    sentences = _load(_RIGHT_PATH)
    failures = [s for s in sentences if not _is_valid(s)]
    passed = len(sentences) - len(failures)
    assert passed >= _RIGHT_VALID_BASELINE, (
        f"Right corpus: only {passed}/{len(sentences)} valid "
        f"(regression: expected >= {_RIGHT_VALID_BASELINE}).\n"
        f"Failures (first 20):\n" + _first(failures)
    )


@pytest.mark.slow
def test_wrong_corpus_rejection_rate():
    """At least _WRONG_BASELINE wrong lines must be rejected (not valid)."""
    sentences = _load(_WRONG_PATH)
    escaped = [s for s in sentences if _is_valid(s)]
    rejected = len(sentences) - len(escaped)
    assert rejected >= _WRONG_BASELINE, (
        f"Wrong corpus: only {rejected}/{len(sentences)} rejected "
        f"(regression: expected >= {_WRONG_BASELINE}).\n"
        f"Escapes (first 20):\n" + _first(escaped)
    )


@pytest.mark.slow
def test_discouraged_corpus_warning_rate():
    """At least _DISCOURAGED_BASELINE discouraged lines must be valid with a WARNING."""
    sentences = _load(_DISCOURAGED_PATH)
    failures = [s for s in sentences if not _is_warned(s)]
    passed = len(sentences) - len(failures)
    assert passed >= _DISCOURAGED_BASELINE, (
        f"Discouraged corpus: only {passed}/{len(sentences)} valid with a warning "
        f"(regression: expected >= {_DISCOURAGED_BASELINE}).\n"
        f"Failures (first 20):\n" + _first(failures)
    )


# ---------------------------------------------------------------------------
# Spot-check: sentences that must always be valid
# ---------------------------------------------------------------------------

_MUST_BE_VALID = [
    # Core sentence types
    "mi pona",
    "mi moku e kasi",
    "jan li wile pali",
    "tomo pi telo nasa",
    "tenpo ni la mi wile toki",
    "jan Joni li pali li toki",
    "sina sona ala sona",
    "mi moku lon tomo",
    "jan pona o toki tawa mi",
    "mi en sina li pona",
    # Preverbs
    "jan li ken moku e kasi suli",
    "ona li wile ala pali",
    "mi kama sona e toki pona",
    # Prepositions
    "mi tawa tomo lon tenpo suno",
    "ona li pana e moku tawa jan",
    # Compound subjects / objects
    "jan en soweli li lon ma",
    "mi moku e kili e telo",
    # Context (la)
    "tenpo pimeja la jan li lape",
    "mi wile moku la mi jo e moku",
    # Proper nouns
    "jan Mewi li pona",
    "ma Tosi li suli",
    # Questions
    "sina moku ala moku",
    "jan li lon seme",
    # Vocative
    "jan o moku",
    "jan pona o kama",
    # Ordinal
    "jan nanpa wan li pona",
    # Sentence starters
    "taso ona li pona",
    "a mi pona",
    # ni as a pronoun (wasona lesson 6)
    "ni li pona",
    "mi wile e ni",
    # Capitalised first word
    "Mi moku.",
]


@pytest.mark.parametrize("sentence", _MUST_BE_VALID)
def test_must_be_valid(sentence):
    """Key sentences that must parse with no ERROR diagnostic."""
    r = validate(sentence)
    assert r.is_valid, f"Expected {sentence!r} to be valid: {r.diagnostics}"


# ---------------------------------------------------------------------------
# Spot-check: specific wrong sentences that must always be rejected
# ---------------------------------------------------------------------------

_MUST_REJECT = [
    # Structural particle in wrong position — parser rejects
    ("e moku lon tomo",        "starts with object marker e"),
    ("li pona",                "starts with predicate marker li"),
    ("pi telo nasa",           "starts with pi"),
    ("la mi moku",             "starts with la"),
    ("mi moku e",              "sentence stops after e"),
    # Phonotactic errors — validator catches
    ("jan Wupi li pona",       "Wu- forbidden syllable onset"),
    ("jan Jipi li pona",       "Ji- forbidden syllable onset"),
    ("jan Tipa li pona",       "Ti- forbidden syllable onset"),
    ("jan Wopi li pona",       "Wo- forbidden syllable onset"),
    ("jan Trump li sona",      "Tr- invalid consonant cluster"),
    # Words outside the dictionary — validator catches
    ("mi moku e xyz",          "lowercase word not in the dictionary"),
]


@pytest.mark.parametrize("sentence,reason", _MUST_REJECT)
def test_must_reject(sentence, reason):
    """Sentences that must always be rejected (not valid)."""
    r = validate(sentence)
    assert not r.is_valid, f"Expected rejection for {sentence!r} ({reason})"

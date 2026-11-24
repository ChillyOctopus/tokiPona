"""
lexer.py
Toki Pona lexer: splits text into sentences and words, and classifies each word.

Run tests with .venv/bin/python3 -m pytest tests/test_lexer.py -v

Token types (checked in priority order):
  PARTICLE     — structural grammar markers: a anu e en la li nanpa o pi
  PREPOSITION  — case/location markers: kepeken lon sama tan tawa
  PREVERB      — auxiliary/modal verbs: awen kama ken lukin open pini sona wile alasa
  NUMBER       — numerals: ala ale ali luka mute tu wan, and wasona's shorthand
                 #WTLMA ("#LTT" = luka tu tu = 9; more/25-01)
  UNMARKED     — 1st/2nd person pronouns that skip 'li': mi sina
  MOD_ONLY     — kin, and proper names (any capitalised word)
  CONTENT      — every other dictionary word (nouns, verbs, modifiers by position)
  UNKNOWN      — lowercase words that are not in the dictionary

Capitals: wasona teaches that sentences start lowercase (lesson 2) and that a
capitalised word is always a name, even one spelled like a dictionary word:
"jan Ali" is a person called Ali, not "everyone" (lesson 7). English habit
still capitalises the first word of a sentence, and a bare name can't start a
sentence (names follow a head noun), so normalize() lowercases a dictionary
word there and nowhere else: "Mi moku." reads mi as the pronoun.

Word sets are derived from master_dictionary.yaml at import time, with small
hardcoded sets for the closed structural classes (particles, prepositions, mi/sina).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("pip install pyyaml")

# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(str, Enum):
    PARTICLE    = "particle"
    PREPOSITION = "preposition"
    PREVERB     = "preverb"
    NUMBER      = "number"
    UNMARKED    = "unmarked_subject"
    MOD_ONLY    = "modifier_only"
    CONTENT     = "content"
    UNKNOWN     = "unknown"


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    type:  TokenType
    text:  str   # lowercase for dictionary words; names keep their capitals
    index: int   # character offset in the original input string


# ---------------------------------------------------------------------------
# Word sets (module-level constants, populated at import time)
# ---------------------------------------------------------------------------

# Structural particles — always PARTICLE regardless of dictionary POS.
PARTICLES: frozenset[str] = frozenset({
    "a", "anu", "e", "en", "la", "li", "nanpa", "o", "pi",
})

# Core prepositions — poka excluded (disputed across sources), la excluded (already PARTICLE).
PREPOSITIONS: frozenset[str] = frozenset({
    "kepeken", "lon", "sama", "tan", "tawa",
})

# Unmarked subjects — 1st/2nd person pronouns that appear without 'li'.
UNMARKED_SUBJECTS: frozenset[str] = frozenset({"mi", "sina"})

# Derived from master_dictionary.yaml by _load_word_sets() below: every
# entry by word, every dictionary word, preverbs (pos "a") and numbers (pos "#").
ENTRIES:    dict[str, dict]
DICTIONARY: frozenset[str]
PREVERBS:   frozenset[str]
NUMBERS:    frozenset[str]


def _load_word_sets() -> tuple[dict[str, dict], frozenset[str], frozenset[str], frozenset[str]]:
    """Load master_dictionary.yaml: every entry, all words, preverbs and numbers."""
    dict_path = Path(__file__).parent / "master_dictionary.yaml"
    with open(dict_path, encoding="utf-8") as fh:
        raw = "\n".join(line for line in fh if not line.startswith("#"))
    words: dict = yaml.safe_load(raw) or {}

    preverbs: set[str] = set()
    numbers:  set[str] = set()

    for word, entry in words.items():
        senses = entry.get("senses", [])
        pos_set = {s.get("pos", "") for s in senses}
        if "a" in pos_set:
            preverbs.add(word)
        if "#" in pos_set:
            numbers.add(word)

    return words, frozenset(words), frozenset(preverbs), frozenset(numbers)


# Initialise at module load.
ENTRIES, DICTIONARY, PREVERBS, NUMBERS = _load_word_sets()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_word(word: str) -> TokenType:
    """
    Return the TokenType for a single word.
    Priority order: MOD_ONLY (names) > PARTICLE > PREPOSITION > PREVERB > NUMBER >
    UNMARKED > MOD_ONLY (kin) > CONTENT > UNKNOWN.
    Classifies a word as it appears after normalize(): a capitalised word is a
    name even if it is spelled like a dictionary word ("Ali"); a lowercase word
    outside the dictionary is UNKNOWN.
    The parser resolves positional ambiguity (e.g. tawa as verb vs preposition).
    """
    if word.startswith("#"):     return TokenType.NUMBER
    if word[:1].isupper():       return TokenType.MOD_ONLY
    lower = word.lower()
    if lower in PARTICLES:       return TokenType.PARTICLE
    if lower in PREPOSITIONS:    return TokenType.PREPOSITION
    if lower in PREVERBS:        return TokenType.PREVERB
    if lower in NUMBERS:         return TokenType.NUMBER
    if lower in UNMARKED_SUBJECTS: return TokenType.UNMARKED
    if lower == "kin":           return TokenType.MOD_ONLY
    if lower in DICTIONARY:      return TokenType.CONTENT
    return TokenType.UNKNOWN


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z]+")

# A token: a word, or a number in wasona's Latin shorthand (W T L M A =
# wan tu luka mute ale, after a #: "tenpo mun #LTT").
_TOKEN_RE = re.compile(r"#[WTLMA]+(?![a-zA-Z])|[a-zA-Z]+")

# A sentence ends at . ! ? or a line break. ':' and ',' stay inside it:
# "ona li toki e ni: ..." and "jan Ken o, ..." are each one sentence.
# A colon stuck to the end mark ("mi moku?:") stays with that sentence, so the
# validator can flag it.
_SENTENCE_END_RE = re.compile(r"[.!?]+:*|\n")

# Words that cannot start a sentence. A line that begins with one carries on
# the line before it (song lyrics: "tenpo pimeja lete / la mi tu li awen seli").
CONTINUATION_WORDS: frozenset[str] = frozenset({"la", "li", "e", "pi"})

# The first word of a sentence: at the start, or after . ! ? : or a line
# break, skipping quotes and other punctuation in between.
_FIRST_WORD_RE = re.compile(r"(?:^|[.!?:\n])[^a-zA-Z]*([a-zA-Z]+)")


def continues_line(previous: str, line: str) -> bool:
    """
    True when a line break between previous and line does not end a sentence:
    line starts with la/li/e/pi, or previous ends with ':' (what it announces
    follows on the next line).
    """
    first = _WORD_RE.match(line.lstrip())
    return (previous.rstrip().endswith(":")
            or bool(first and first.group().lower() in CONTINUATION_WORDS))


def normalize(text: str) -> str:
    """
    Lowercase a dictionary word that starts a sentence ("Mi moku." → "mi moku.").
    Every other capitalised word is a name and keeps its capital ("jan Ali").
    Only ASCII letters change case, so every character keeps its offset.
    """
    chars = list(text)
    for m in _FIRST_WORD_RE.finditer(text):
        word = m.group(1)
        if word[0].isupper() and word.lower() in DICTIONARY:
            chars[m.start(1):m.end(1)] = word.lower()
    return "".join(chars)


def split_sentences(text: str) -> list[tuple[int, str]]:
    """
    Split text into sentences: (offset, sentence) pairs, where offset is the
    sentence's position in text and the sentence keeps its end punctuation.
    A line break ends a sentence unless continues_line() says otherwise.
    Pieces without any letters (stray punctuation) are dropped.
    """
    pieces: list[tuple[int, str]] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        if m.group() == "\n" and continues_line(text[start:m.start()], text[m.end():]):
            continue
        pieces.append((start, text[start:m.end()]))
        start = m.end()
    pieces.append((start, text[start:]))

    return [
        (offset + len(piece) - len(piece.lstrip()), piece.strip())
        for offset, piece in pieces
        if _WORD_RE.search(piece)
    ]


def tokenize(text: str) -> list[Token]:
    """
    Tokenize a Toki Pona string.
    All non-alphabetic characters (punctuation, whitespace) are skipped.
    A sentence's first word comes back lowercase if it is a dictionary word;
    every other capitalised word is a name.
    Token.index is the character offset of the word in the original string.
    """
    return [
        Token(type=classify_word(m.group()), text=m.group(), index=m.start())
        for m in _TOKEN_RE.finditer(normalize(text))
    ]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _DEMO = [
        "mi moku e kasi.",
        "jan Joni li wile lukin e sitelen.",
        "sina pali ala pali?",
        "tomo pi telo nasa",
        "ona li tawa tomo kepeken noka.",
        "mi sona pali e ilo.",
    ]

    for sentence in _DEMO:
        print(f"\n  {sentence!r}")
        for tok in tokenize(sentence):
            print(f"    [{tok.type.value:<18}] {tok.text!r:20} @{tok.index}")
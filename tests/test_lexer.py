"""
tests/test_lexer.py
Unit tests for the Toki Pona lexer.
Run with: python -m pytest tests/test_lexer.py -v
"""

import sys
from pathlib import Path

# Allow importing lexer from the project root when running from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexer import (
    Token, TokenType,
    DICTIONARY, ENTRIES, PARTICLES, PREPOSITIONS, PREVERBS, NUMBERS, UNMARKED_SUBJECTS,
    classify_word, normalize, split_sentences, tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def types(text: str) -> list[str]:
    """Return just the token type values for a sentence."""
    return [t.type.value for t in tokenize(text)]


def words_and_types(text: str) -> list[tuple[str, str]]:
    return [(t.text, t.type.value) for t in tokenize(text)]


# ---------------------------------------------------------------------------
# Word set smoke tests
# ---------------------------------------------------------------------------

def test_word_sets_non_empty():
    assert len(PARTICLES) >= 9
    assert len(PREPOSITIONS) == 5
    assert len(PREVERBS) >= 7
    assert len(NUMBERS) >= 6
    assert len(UNMARKED_SUBJECTS) == 2
    assert len(DICTIONARY) >= 120


def test_particle_set():
    for w in ("li", "e", "pi", "la", "en", "o", "anu", "a", "nanpa"):
        assert w in PARTICLES, f"'{w}' should be in PARTICLES"


def test_preposition_set():
    for w in ("kepeken", "lon", "sama", "tan", "tawa"):
        assert w in PREPOSITIONS, f"'{w}' should be in PREPOSITIONS"


def test_preverb_set():
    for w in ("wile", "ken", "kama", "awen", "lukin", "sona", "alasa"):
        assert w in PREVERBS, f"'{w}' should be in PREVERBS"


def test_number_set():
    for w in ("wan", "tu", "luka", "mute", "ale", "ali", "ala"):
        assert w in NUMBERS, f"'{w}' should be in NUMBERS"


# ---------------------------------------------------------------------------
# classify_word
# ---------------------------------------------------------------------------

def test_classify_particles():
    for w in PARTICLES:
        assert classify_word(w) == TokenType.PARTICLE

def test_classify_prepositions():
    for w in PREPOSITIONS:
        assert classify_word(w) == TokenType.PREPOSITION

def test_classify_preverbs():
    for w in ("wile", "ken", "kama", "awen", "lukin", "sona", "alasa"):
        assert classify_word(w) == TokenType.PREVERB, f"'{w}' should be PREVERB"

def test_classify_numbers():
    for w in ("wan", "tu", "luka", "mute", "ale", "ala"):
        assert classify_word(w) == TokenType.NUMBER, f"'{w}' should be NUMBER"

def test_classify_unmarked_subjects():
    assert classify_word("mi")   == TokenType.UNMARKED
    assert classify_word("sina") == TokenType.UNMARKED

def test_classify_kin():
    assert classify_word("kin") == TokenType.MOD_ONLY

def test_classify_proper_noun():
    # Capitalised words outside the dictionary are names
    assert classify_word("Joni")   == TokenType.MOD_ONLY
    assert classify_word("Mewi")   == TokenType.MOD_ONLY
    assert classify_word("Inli")   == TokenType.MOD_ONLY

def test_classify_content_words():
    for w in ("jan", "tomo", "pona", "moku", "soweli", "pipi", "ona"):
        assert classify_word(w) == TokenType.CONTENT, f"'{w}' should be CONTENT"

def test_classify_capitalised_word_is_a_name():
    # A capital marks a name even on a dictionary spelling ("jan Ali");
    # normalize() has already lowercased a sentence's first word.
    for w in ("Ali", "Toki", "Mi", "LI"):
        assert classify_word(w) == TokenType.MOD_ONLY, w

def test_number_shorthand_is_a_number():
    # wasona more/25-01: "#LTT" = luka tu tu
    assert classify_word("#LTT") == TokenType.NUMBER
    assert words_and_types("tenpo mun #LTT la") == [
        ("tenpo", "content"), ("mun", "content"), ("#LTT", "number"), ("la", "particle"),
    ]

def test_classify_unknown_word():
    assert classify_word("xyz")   == TokenType.UNKNOWN
    assert classify_word("hello") == TokenType.UNKNOWN


# ---------------------------------------------------------------------------
# tokenize — sentence-level tests
# ---------------------------------------------------------------------------

def test_basic_transitive():
    # mi moku e kasi
    result = words_and_types("mi moku e kasi")
    assert result == [
        ("mi",   "unmarked_subject"),
        ("moku", "content"),
        ("e",    "particle"),
        ("kasi", "content"),
    ]

def test_marked_subject_predicate():
    # jan li pali
    result = words_and_types("jan li pali")
    assert result == [
        ("jan",  "content"),
        ("li",   "particle"),
        ("pali", "content"),
    ]

def test_preverb_phrase():
    # jan li wile pali
    result = words_and_types("jan li wile pali")
    assert result == [
        ("jan",  "content"),
        ("li",   "particle"),
        ("wile", "preverb"),
        ("pali", "content"),
    ]

def test_sona_as_preverb():
    # mi sona pali e ilo  (sona = "to know how to")
    result = words_and_types("mi sona pali e ilo")
    assert result[1] == ("sona", "preverb")

def test_alasa_as_preverb():
    # ona li alasa kama
    result = words_and_types("ona li alasa kama")
    assert result[2] == ("alasa", "preverb")

def test_preposition_phrase():
    # ona li tawa tomo kepeken noka
    result = words_and_types("ona li tawa tomo kepeken noka")
    assert result == [
        ("ona",     "content"),
        ("li",      "particle"),
        ("tawa",    "preposition"),
        ("tomo",    "content"),
        ("kepeken", "preposition"),
        ("noka",    "content"),
    ]

def test_pi_phrase():
    # tomo pi telo nasa
    result = words_and_types("tomo pi telo nasa")
    assert result == [
        ("tomo",  "content"),
        ("pi",    "particle"),
        ("telo",  "content"),
        ("nasa",  "content"),
    ]

def test_proper_noun_in_sentence():
    # jan Joni li pona
    result = words_and_types("jan Joni li pona")
    assert result == [
        ("jan",  "content"),
        ("Joni", "modifier_only"),
        ("li",   "particle"),
        ("pona", "content"),
    ]

def test_number_as_modifier():
    # jan tu li pali
    result = words_and_types("jan tu li pali")
    assert result[1] == ("tu", "number")

def test_ala_as_number():
    # X ala X question — ala is NUMBER; parser handles negation context
    result = words_and_types("sina sona ala sona")
    assert result == [
        ("sina", "unmarked_subject"),
        ("sona", "preverb"),
        ("ala",  "number"),
        ("sona", "preverb"),
    ]

def test_punctuation_stripped():
    # punctuation and whitespace are discarded
    tokens = tokenize("pona! ike.")
    assert [(t.text, t.type.value) for t in tokens] == [
        ("pona", "content"),
        ("ike",  "content"),
    ]

def test_empty_string():
    assert tokenize("") == []

def test_punctuation_only():
    assert tokenize("... !!! ???") == []

def test_index_positions():
    # "mi li pona" — check character offsets
    tokens = tokenize("mi li pona")
    assert tokens[0].index == 0   # "mi"
    assert tokens[1].index == 3   # "li"
    assert tokens[2].index == 6   # "pona"

def test_multisentence():
    # Two sentences — all tokens present, punctuation skipped
    result = types("mi pona. sina ike.")
    assert result == [
        "unmarked_subject", "content",
        "unmarked_subject", "content",
    ]

def test_kin_modifier():
    # kin should be MOD_ONLY
    result = words_and_types("pona kin")
    assert result[1] == ("kin", "modifier_only")

def test_la_is_particle():
    # la is always PARTICLE (context boundary), not preposition
    assert classify_word("la") == TokenType.PARTICLE
    result = words_and_types("tenpo ni la mi pona")
    la_tok = [t for t in tokenize("tenpo ni la mi pona") if t.text == "la"]
    assert la_tok[0].type == TokenType.PARTICLE

def test_vocative():
    # jan pona o toki! → CONTENT CONTENT PARTICLE CONTENT
    result = types("jan pona o toki")
    assert result == ["content", "content", "particle", "content"]

def test_complex_sentence():
    # jan Joni li wile lukin e sitelen
    result = words_and_types("jan Joni li wile lukin e sitelen")
    assert result == [
        ("jan",     "content"),
        ("Joni",    "modifier_only"),
        ("li",      "particle"),
        ("wile",    "preverb"),
        ("lukin",   "preverb"),
        ("e",       "particle"),
        ("sitelen", "content"),
    ]

def test_token_is_immutable():
    tok = tokenize("pona")[0]
    try:
        tok.text = "ike"
        assert False, "Token should be frozen (immutable)"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# normalize / split_sentences
# ---------------------------------------------------------------------------

def test_first_word_lowercased_names_kept():
    assert words_and_types("Mi moku. Jan Ali li pona!") == [
        ("mi",   "unmarked_subject"),
        ("moku", "content"),
        ("jan",  "content"),
        ("Ali",  "modifier_only"),
        ("li",   "particle"),
        ("pona", "content"),
    ]

def test_normalize_only_touches_first_words():
    # Sentence starts: the text start, after . ! ? : and line breaks (quotes skipped)
    text = 'Mi moku. Jan Ali li pona: "Ona li kama."\nLa mi lape.'
    assert normalize(text) == 'mi moku. jan Ali li pona: "ona li kama."\nla mi lape.'

def test_split_sentences():
    assert split_sentences("mi moku. sina pali! ona li seme?") == [
        (0, "mi moku."), (9, "sina pali!"), (20, "ona li seme?"),
    ]

def test_split_keeps_colon_and_comma_inside():
    text = "jan Ken o, ona li toki e ni: mi pona."
    assert split_sentences(text) == [(0, text)]

def test_split_on_line_breaks():
    assert split_sentences("mi moku\nsina pali") == [(0, "mi moku"), (8, "sina pali")]

def test_split_joins_continuation_lines():
    # A line starting with la/li/e/pi carries on the one before (basics-3/29-song)
    text = "tenpo pimeja lete\nla mi tu li awen seli"
    assert split_sentences(text) == [(0, text)]

def test_split_joins_line_after_colon():
    # What a colon announces can follow on the next line (basics-1/02-writing)
    text = "taso sina sona ala:\nutala li ike e sina."
    assert split_sentences(text) == [(0, text)]

def test_split_keeps_a_colon_on_its_end_mark():
    # "?:" stays in one sentence so the validator can flag it
    assert split_sentences("mi moku?: mi lape.") == [(0, "mi moku?:"), (10, "mi lape.")]

def test_split_drops_pieces_without_words():
    assert split_sentences("... mi pona!!! ?") == [(4, "mi pona!!!")]

def test_wrapped_dictionary_lines_are_not_words():
    # dictionary.txt wraps some glosses onto a new line ("manage to, start to"
    # under kama); build_dictionary.py once read those lines as words
    assert "manage" not in DICTIONARY
    assert [s["gloss"] for s in ENTRIES["n"]["senses"]] == ["hmm, umm"]

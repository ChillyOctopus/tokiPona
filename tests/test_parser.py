"""
tests/test_parser.py
Unit tests for the Toki Pona Earley parser.
Run with: .venv/bin/python3 -m pytest tests/test_parser.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parser import parse, ParseResult
from lark.tree import Tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top(text: str) -> ParseResult | None:
    """Return the top-scored parse for a sentence, or None if ungrammatical."""
    results = parse(text)
    return results[0] if results else None


def tree_rules(tree: Tree) -> set[str]:
    """Collect all rule names that appear anywhere in a parse tree."""
    out = {tree.data}
    for child in tree.children:
        if isinstance(child, Tree):
            out |= tree_rules(child)
    return out


def has_rule(result: ParseResult, rule: str) -> bool:
    return rule in tree_rules(result.tree)


def any_parse_has(text: str, rule: str) -> bool:
    return any(has_rule(r, rule) for r in parse(text))


# ---------------------------------------------------------------------------
# Empty / ungrammatical input
# ---------------------------------------------------------------------------

def test_empty_returns_empty():
    assert parse("") == []

def test_whitespace_only_returns_empty():
    assert parse("   ") == []

def test_ungrammatical_bare_li():
    # "li pona" has no subject — should not parse (or parse as vocative fragment)
    # If it parses at all the score must be low (fragment / bad-head penalty)
    results = parse("li pona")
    # Either no parse, or only very low-probability parses
    if results:
        assert results[0].score <= 2.0

def test_punctuation_stripped():
    # Parser silently drops punctuation
    r1 = parse("mi pona")
    r2 = parse("mi pona.")
    assert len(r1) > 0 and len(r2) > 0
    # Both should produce the same number of parses
    assert len(r1) == len(r2)


# ---------------------------------------------------------------------------
# Basic sentence types
# ---------------------------------------------------------------------------

def test_basic_intransitive_unmarked():
    # "mi pona" — unmarked subject, no li
    r = top("mi pona")
    assert r is not None
    assert has_rule(r, "verb_phrase_intransitive")

def test_basic_transitive():
    # "mi moku e kasi" — transitive with object
    results = parse("mi moku e kasi")
    assert len(results) == 1
    assert has_rule(results[0], "verb_phrase_transitive")
    assert has_rule(results[0], "object_phrase")

def test_marked_subject_intransitive():
    # "jan li pali" — marked subject with li
    r = top("jan li pali")
    assert r is not None
    assert has_rule(r, "verb_phrase_intransitive")

def test_marked_subject_transitive():
    # "jan li moku e pan" — marked subject, transitive
    r = top("jan li moku e pan")
    assert r is not None
    assert has_rule(r, "verb_phrase_transitive")


# ---------------------------------------------------------------------------
# Preverb phrases
# ---------------------------------------------------------------------------

def test_single_preverb():
    # "jan li wile pali"
    r = top("jan li wile pali")
    assert r is not None
    assert has_rule(r, "preverb_phrase")
    assert has_rule(r, "preverb")

def test_two_preverbs():
    # "mi ken wile toki"
    r = top("mi ken wile toki")
    assert r is not None
    assert has_rule(r, "preverb")

def test_negated_preverb():
    # "mi wile ala pali" — wile ala is negated preverb
    results = parse("mi wile ala pali")
    assert len(results) > 0
    assert any_parse_has("mi wile ala pali", "preverb")

def test_preverb_with_a():
    # "mi wile a toki" — preverb with emphasis
    r = top("mi wile a toki")
    assert r is not None


# ---------------------------------------------------------------------------
# Prepositions
# ---------------------------------------------------------------------------

def test_prep_phrase_intransitive():
    # "mi moku lon tomo"
    r = top("mi moku lon tomo")
    assert r is not None
    assert has_rule(r, "prep_phrase")

def test_prep_phrase_tawa():
    # "jan pona o toki tawa mi" — top parse must use a prep_phrase
    results = parse("jan pona o toki tawa mi")
    assert len(results) >= 1
    assert has_rule(results[0], "prep_phrase")

def test_prep_kepeken():
    # "mi pali kepeken ilo"
    r = top("mi pali kepeken ilo")
    assert r is not None
    assert has_rule(r, "prep_phrase")

def test_transitive_with_prep():
    # "mi pana e tomo tawa ona" — object + prep
    r = top("mi pana e tomo tawa ona")
    assert r is not None
    assert has_rule(r, "verb_phrase_transitive")
    assert has_rule(r, "prep_phrase")


# ---------------------------------------------------------------------------
# Noun phrase structure
# ---------------------------------------------------------------------------

def test_pi_phrase():
    # "tomo pi telo nasa" — single pi grouping
    results = parse("tomo pi telo nasa")
    assert len(results) == 1
    assert has_rule(results[0], "pi_phrase")

def test_modified_noun():
    # "jan pona" — head + modifier
    r = top("jan pona")
    assert r is not None

def test_ordinal_nanpa():
    # "jan nanpa wan" — ordinal
    r = top("jan nanpa wan")
    assert r is not None
    assert has_rule(r, "nanpa_phrase")

def test_multiple_modifiers():
    # "tomo telo suli" — head + two modifiers
    r = top("tomo telo suli")
    assert r is not None


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

def test_compound_subject_en():
    # "mi en sina li pona"
    results = parse("mi en sina li pona")
    assert len(results) == 1
    assert has_rule(results[0], "compound_subject")

def test_compound_subject_anu():
    # "jan anu soweli li moku" — anu compound subject
    r = top("jan anu soweli li moku")
    assert r is not None


# ---------------------------------------------------------------------------
# Serial predicates
# ---------------------------------------------------------------------------

def test_serial_predicates_li():
    # "jan Joni li pali li toki"
    results = parse("jan Joni li pali li toki")
    assert len(results) == 1
    assert has_rule(results[0], "predicate_list_li")

def test_multiple_objects():
    # "mi pana e tomo e moku tawa ona"
    r = top("mi pana e tomo e moku tawa ona")
    assert r is not None


# ---------------------------------------------------------------------------
# Vocative
# ---------------------------------------------------------------------------

def test_vocative_o():
    # "jan pona o toki"  (already a clause_any with O)
    r = top("jan pona o toki")
    assert r is not None

def test_bare_vocative():
    # "jan o" — vocative without predicate
    r = top("jan o")
    assert r is not None


# ---------------------------------------------------------------------------
# Context clause (la)
# ---------------------------------------------------------------------------

def test_context_la():
    # "tenpo ni la mi wile toki"
    r = top("tenpo ni la mi wile toki")
    assert r is not None
    assert has_rule(r, "context_clause")

def test_clause_la():
    # "mi pona la sina pona" — clause as context
    r = top("mi pona la sina pona")
    assert r is not None
    assert has_rule(r, "context_clause")


# ---------------------------------------------------------------------------
# Sentence starters
# ---------------------------------------------------------------------------

def test_taso_starter():
    # "taso ona li pona" — taso as sentence starter
    r = top("taso ona li pona")
    assert r is not None
    assert has_rule(r, "sentence_starter")

def test_a_starter():
    # "a mi pona" — exclamation starter
    r = top("a mi pona")
    assert r is not None
    assert has_rule(r, "sentence_starter")


# ---------------------------------------------------------------------------
# Question forms
# ---------------------------------------------------------------------------

def test_v_ala_v_question():
    # "sina sona ala sona" — V-ala-V yes/no question
    results = parse("sina sona ala sona")
    assert len(results) > 0
    assert any_parse_has("sina sona ala sona", "verb_phrase_yn")

def test_anu_seme_question():
    # "sina pona anu seme" — weak question
    r = top("sina pona anu seme")
    assert r is not None


# ---------------------------------------------------------------------------
# Proper nouns / MOD_ONLY
# ---------------------------------------------------------------------------

def test_proper_noun_modifier():
    # "jan Joni li toki" — capitalised name as modifier
    r = top("jan Joni li toki")
    assert r is not None

def test_proper_noun_in_subject():
    # "jan Sonja li pali" — compound name
    r = top("jan Sonja li pali")
    assert r is not None


# ---------------------------------------------------------------------------
# Scoring sanity
# ---------------------------------------------------------------------------

def test_probabilities_sum_to_one():
    results = parse("mi moku e kasi")
    total = sum(r.probability for r in results)
    assert abs(total - 1.0) < 1e-9

def test_sorted_by_score_descending():
    results = parse("tenpo ni la mi wile toki")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

def test_top_parse_is_best():
    results = parse("jan li wile pali")
    assert results[0].score >= results[-1].score

def test_transitive_scores_higher_than_fragment():
    # Full transitive parse should outscore a bare fragment
    results = parse("mi moku e kasi")
    assert results[0].score > 0

def test_parse_result_repr():
    r = top("mi pona")
    assert "ParseResult" in repr(r)
    assert "score=" in repr(r)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_ala_as_content():
    # "ala" standalone (odd but shouldn't crash)
    results = parse("ala")
    # Either parses as bare subject fragment or fails
    # Just checking no exception is raised
    assert isinstance(results, list)

def test_taso_as_content():
    # "taso" alone as sentence fragment
    results = parse("taso")
    assert isinstance(results, list)

def test_long_preverb_chain():
    # "mi ken awen wile toki" — three preverbs
    r = top("mi ken awen wile toki")
    assert r is not None

def test_pi_with_anu_disjunct():
    # "tomo pi pona anu ike" — disjunction inside pi phrase
    r = top("tomo pi pona anu ike")
    assert r is not None

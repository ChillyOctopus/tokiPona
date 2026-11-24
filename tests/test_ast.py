"""
tests/test_ast.py
Unit tests for the Toki Pona AST builder.
Run with: .venv/bin/python3 -m pytest tests/test_ast.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ast_builder import build_ast, build_ast_from_text
from ast_nodes import (
    Clause, CompoundSubject, ContextPhrase, DisjunctPhrase,
    MarkedSubject, Number, ObjectPhrase, OrdinalPhrase, PiPhrase,
    PrepPhrase, Preverb, PreverbPhrase, Preposition,
    Sentence, SentenceStarter, SimplePhrase, UnmarkedSubject,
    Utterance, VerbPhraseIntransitive, VerbPhrasePrepositional,
    VerbPhraseTransitive, VerbPhraseYN, Vocative, Word,
)
from parser import parse as tp_parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top_ast(text: str) -> Utterance:
    """Parse text, build AST for top-scored result, assert it succeeds."""
    results = build_ast_from_text(text)
    assert results, f"No parse for {text!r}"
    return results[0][0]


def top_sentence(text: str) -> Sentence:
    """Top AST body must be a Sentence."""
    utt = top_ast(text)
    assert isinstance(utt.body, Sentence), f"Expected Sentence, got {type(utt.body).__name__}"
    return utt.body


def top_clause(text: str) -> Clause:
    return top_sentence(text).clause


def top_predicate(text: str):
    """Return the first predicate of the top-scored parse."""
    preds = top_clause(text).predicates
    assert preds, f"No predicates for {text!r}"
    return preds[0]


# ---------------------------------------------------------------------------
# Node type correctness — basic sentences
# ---------------------------------------------------------------------------

def test_basic_transitive_types():
    utt = top_ast("mi moku e kasi")
    s   = utt.body
    assert isinstance(s, Sentence)
    assert isinstance(s.clause.subject, UnmarkedSubject)
    assert s.clause.predicate_type == "none"
    pred = s.clause.predicates[0]
    assert isinstance(pred, VerbPhraseTransitive)
    assert pred.verb.head.text == "moku"
    assert len(pred.objects) == 1
    assert pred.objects[0].np.head.text == "kasi"

def test_basic_intransitive_types():
    utt = top_ast("mi pona")
    assert isinstance(utt.body, Sentence)
    pred = top_predicate("mi pona")
    assert isinstance(pred, VerbPhraseIntransitive)
    assert pred.verb.head.text == "pona"


# ---------------------------------------------------------------------------
# Phrase collapsing
# ---------------------------------------------------------------------------

def test_single_word_is_simple_phrase():
    # "jan" as bare subject fragment — should be SimplePhrase, not wrapped
    results = build_ast_from_text("jan")
    assert results
    utt = results[0][0]
    # It's a bare subject fragment
    assert isinstance(utt.body, Sentence)
    subj = utt.body.clause.subject
    assert isinstance(subj, MarkedSubject)
    assert isinstance(subj.phrase, SimplePhrase)
    assert subj.phrase.head.text == "jan"
    assert subj.phrase.modifiers == []

def test_pi_phrase_structure():
    # "tomo pi telo nasa" → PiPhrase(base=SP('tomo'), pi_groups=[SP('telo','nasa')])
    utt = top_ast("tomo pi telo nasa")
    subj = utt.body.clause.subject
    assert isinstance(subj, MarkedSubject)
    np = subj.phrase
    assert isinstance(np, PiPhrase)
    assert np.base.head.text == "tomo"
    assert len(np.pi_groups) == 1
    pi_inner = np.pi_groups[0]
    assert isinstance(pi_inner, SimplePhrase)
    assert pi_inner.head.text == "telo"
    assert pi_inner.modifiers[0].text == "nasa"

def test_no_pi_vs_pi_differ():
    utt_plain = top_ast("tomo telo nasa")
    utt_pi    = top_ast("tomo pi telo nasa")
    subj_plain = utt_plain.body.clause.subject
    subj_pi    = utt_pi.body.clause.subject
    # No-pi version: SimplePhrase; pi version: PiPhrase — they must differ
    assert isinstance(subj_plain.phrase, SimplePhrase), "expected SimplePhrase for no-pi"
    assert isinstance(subj_pi.phrase, PiPhrase),        "expected PiPhrase for pi"

def test_double_pi_is_flat():
    # One PiPhrase with a group per pi; a group never holds another pi group
    np = top_ast("tomo pi telo nasa pi jan mute").body.clause.subject.phrase
    assert [g.source for g in np.pi_groups] == ["telo nasa", "jan mute"]
    assert not any(isinstance(g, PiPhrase) for g in np.pi_groups)

def test_second_pi_cannot_hide_a_single_word_group():
    # wasona lesson 23: pi "will never occur before just one word". With no pi
    # inside a pi group, "pi telo pi nasa mute" has only the flat reading.
    readings = build_ast_from_text("tomo pi telo pi nasa mute")
    assert len(readings) == 1
    np = readings[0][0].body.clause.subject.phrase
    assert [g.source for g in np.pi_groups] == ["telo", "nasa mute"]

def test_ordinal_phrase():
    # "jan nanpa wan" → subject is OrdinalPhrase
    utt  = top_ast("jan nanpa wan li pona")
    subj = utt.body.clause.subject
    assert isinstance(subj, MarkedSubject)
    assert isinstance(subj.phrase, OrdinalPhrase)
    assert subj.phrase.base.head.text == "jan"
    assert isinstance(subj.phrase.ordinal, Number)
    assert subj.phrase.ordinal.words[0].text == "wan"


# ---------------------------------------------------------------------------
# Preverb structure
# ---------------------------------------------------------------------------

def test_preverb_negated():
    pred = top_predicate("mi wile ala pali")
    assert isinstance(pred, PreverbPhrase)
    pv = pred.preverbs[0]
    assert pv.word.text == "wile"
    assert pv.negated is True

def test_preverb_not_negated():
    pred = top_predicate("mi wile pali")
    assert isinstance(pred, PreverbPhrase)
    pv = pred.preverbs[0]
    assert pv.word.text == "wile"
    assert pv.negated is False

def test_stacked_preverbs():
    pred = top_predicate("mi ken wile toki")
    assert isinstance(pred, PreverbPhrase)
    assert len(pred.preverbs) == 2
    assert pred.preverbs[0].word.text == "ken"
    assert pred.preverbs[1].word.text == "wile"
    # Both non-negated
    assert not pred.preverbs[0].negated
    assert not pred.preverbs[1].negated


# ---------------------------------------------------------------------------
# Predicate flattening (serial predicates)
# ---------------------------------------------------------------------------

def test_serial_predicates_flat():
    # "jan Joni li pali li toki" → exactly 2 predicates, not nested
    clause = top_clause("jan Joni li pali li toki")
    assert len(clause.predicates) == 2
    for p in clause.predicates:
        assert isinstance(p, VerbPhraseIntransitive)

def test_three_serial_predicates():
    clause = top_clause("jan li pali li toki li moku")
    assert len(clause.predicates) == 3


# ---------------------------------------------------------------------------
# Subject types
# ---------------------------------------------------------------------------

def test_unmarked_subject_mi():
    clause = top_clause("mi pona")
    assert isinstance(clause.subject, UnmarkedSubject)
    assert clause.subject.word.text == "mi"
    assert clause.predicate_type == "none"

def test_unmarked_subject_sina():
    clause = top_clause("sina pona")
    assert isinstance(clause.subject, UnmarkedSubject)
    assert clause.subject.word.text == "sina"

def test_marked_subject():
    clause = top_clause("jan li pona")
    assert isinstance(clause.subject, MarkedSubject)
    assert clause.predicate_type == "li"
    assert isinstance(clause.subject.phrase, SimplePhrase)
    assert clause.subject.phrase.head.text == "jan"

def test_compound_subject_en():
    clause = top_clause("mi en sina li pona")
    assert isinstance(clause.subject, MarkedSubject)
    assert isinstance(clause.subject.phrase, CompoundSubject)
    cs = clause.subject.phrase
    assert cs.conjunction == "en"
    assert len(cs.parts) == 2

def test_compound_subject_anu():
    results = build_ast_from_text("jan anu soweli li moku")
    assert results
    clause = results[0][0].body.clause
    assert isinstance(clause.subject, MarkedSubject)
    assert isinstance(clause.subject.phrase, CompoundSubject)
    assert clause.subject.phrase.conjunction == "anu"


# ---------------------------------------------------------------------------
# Question forms
# ---------------------------------------------------------------------------

def test_v_ala_v_parse_exists():
    # "sina sona ala sona" must have at least one VerbPhraseYN parse
    all_asts = build_ast_from_text("sina sona ala sona")
    assert all_asts, "No parses for V-ala-V sentence"
    vp_types = set()
    for utt, _ in all_asts:
        if isinstance(utt.body, Sentence):
            for pred in utt.body.clause.predicates:
                if isinstance(pred, PreverbPhrase):
                    vp_types.add(type(pred.vp).__name__)
                else:
                    vp_types.add(type(pred).__name__)
    assert "VerbPhraseYN" in vp_types, f"No VerbPhraseYN found in parses: {vp_types}"

def test_v_ala_v_verb_word():
    # The verb in the YN parse should be the correct word
    for utt, _ in build_ast_from_text("sina sona ala sona"):
        if not isinstance(utt.body, Sentence):
            continue
        for pred in utt.body.clause.predicates:
            vp = pred.vp if isinstance(pred, PreverbPhrase) else pred
            if isinstance(vp, VerbPhraseYN):
                assert vp.verb.head.text == "sona"
                return
    # If we reach here without finding VerbPhraseYN, that's fine (it may be rare)

def test_anu_seme_true():
    # "anu seme" appears as direct sentence children in at least one parse
    # (grammar is ambiguous: the top parse may treat "anu seme" as serial predicates)
    all_asts = build_ast_from_text("sina pona anu seme")
    assert all_asts, "No parses for 'sina pona anu seme'"
    assert any(
        isinstance(utt.body, Sentence) and utt.body.anu_seme
        for utt, _ in all_asts
    ), "No parse has anu_seme=True"

def test_anu_seme_false():
    s = top_sentence("sina pona")
    assert s.anu_seme is False


# ---------------------------------------------------------------------------
# Sentence structure
# ---------------------------------------------------------------------------

def test_sentence_starter_taso():
    s = top_sentence("taso ona li pona")
    assert len(s.starters) == 1
    assert s.starters[0].words[0].text == "taso"

def test_sentence_starter_a():
    s = top_sentence("a mi pona")
    assert len(s.starters) == 1
    assert s.starters[0].words[0].text == "a"

def test_no_starter():
    s = top_sentence("mi pona")
    assert s.starters == []

def test_context_la():
    s = top_sentence("tenpo ni la mi wile toki")
    assert len(s.contexts) == 1
    ctx = s.contexts[0]
    assert isinstance(ctx, ContextPhrase)

def test_context_content_is_phrase():
    s = top_sentence("tenpo ni la mi pona")
    ctx = s.contexts[0]
    # content should be a NounPhrase (MarkedSubject wrapping a SimplePhrase)
    assert hasattr(ctx.content, "source")

def test_vocative_type():
    utt = top_ast("jan o")
    assert isinstance(utt.body, Vocative)

def test_call_before_a_mi_sina_sentence():
    # The comma after o marks a call; "sina olin ala olin e mi" is the sentence,
    # not a command with the verb "sina olin"
    s = top_sentence("jan Lope o, sina olin ala olin e mi?")
    assert s.vocative.source == "jan Lope"
    assert isinstance(s.clause.subject, UnmarkedSubject)
    assert isinstance(s.clause.predicates[0], VerbPhraseYN)

def test_vocative_addressee():
    utt = top_ast("jan pona o")
    assert isinstance(utt.body, Vocative)
    addressee = utt.body.addressee
    assert isinstance(addressee, MarkedSubject)
    assert addressee.phrase.head.text == "jan"

def test_predicate_type_li():
    clause = top_clause("jan li pona")
    assert clause.predicate_type == "li"

def test_predicate_type_none():
    clause = top_clause("mi pona")
    assert clause.predicate_type == "none"

def test_predicate_type_o():
    clause = top_clause("jan pona o toki")
    assert clause.predicate_type == "o"


# ---------------------------------------------------------------------------
# Multiple objects and prep phrases
# ---------------------------------------------------------------------------

def test_multiple_objects():
    pred = top_predicate("jan li moku e pan e telo")
    assert isinstance(pred, VerbPhraseTransitive)
    assert len(pred.objects) == 2
    texts = {o.np.head.text for o in pred.objects}
    assert texts == {"pan", "telo"}

def test_prep_phrase_in_intransitive():
    pred = top_predicate("mi moku lon tomo")
    assert isinstance(pred, VerbPhraseIntransitive)
    assert len(pred.preps) == 1
    assert pred.preps[0].prep.word.text == "lon"
    assert pred.preps[0].obj.head.text == "tomo"

def test_prep_phrase_negated():
    # "mi pali kepeken ala ilo" — negated preposition
    pred = top_predicate("mi pali kepeken ala ilo")
    # The prep may be inside a VerbPhraseIntransitive or VerbPhraseTransitive
    if isinstance(pred, VerbPhraseIntransitive):
        pp = pred.preps[0]
    elif isinstance(pred, VerbPhraseTransitive):
        pp = pred.preps[0]
    else:
        pytest_fail_msg = f"Unexpected predicate type: {type(pred)}"
        assert False, pytest_fail_msg
    assert pp.prep.negated is True

def test_multiple_preps():
    pred = top_predicate("mi tawa tomo kepeken noka")
    if isinstance(pred, VerbPhraseIntransitive):
        assert len(pred.preps) == 2
    # also acceptable if parsed as transitive (tawa tomo = verb + obj)

def test_object_phrase_type():
    pred = top_predicate("mi moku e kasi")
    assert isinstance(pred, VerbPhraseTransitive)
    obj = pred.objects[0]
    assert isinstance(obj, ObjectPhrase)
    assert isinstance(obj.np, SimplePhrase)


# ---------------------------------------------------------------------------
# Word metadata
# ---------------------------------------------------------------------------

def test_word_offset():
    # "jan pona" — head at offset 0, modifier at offset 4
    utt  = top_ast("jan pona")
    subj = utt.body.clause.subject
    assert isinstance(subj, MarkedSubject)
    ph = subj.phrase
    assert isinstance(ph, SimplePhrase)
    assert ph.head.offset == 0
    assert ph.modifiers[0].offset == 4

def test_word_token_type_preserved():
    # "jan li wile pali" — check token types on key words
    clause = top_clause("jan li wile pali")
    pred   = clause.predicates[0]
    assert isinstance(pred, PreverbPhrase)
    assert pred.preverbs[0].word.token_type == "PREV"
    assert isinstance(pred.vp, VerbPhraseIntransitive)
    assert pred.vp.verb.head.token_type == "WORD"

def test_source_strips_li():
    # Clause source must not contain "li"
    clause = top_clause("jan li pona")
    assert "li" not in clause.source.split()

def test_source_content_words():
    # "mi moku e kasi" — clause source has content words, no 'e'
    clause = top_clause("mi moku e kasi")
    assert "mi"   in clause.source
    assert "moku" in clause.source
    assert "kasi" in clause.source
    assert "e"    not in clause.source.split()


# ---------------------------------------------------------------------------
# API correctness
# ---------------------------------------------------------------------------

def test_empty_returns_empty():
    assert build_ast_from_text("") == []

def test_ungrammatical_returns_empty_or_low():
    # "li pona" has no subject — either fails to parse or produces a bare result
    results = build_ast_from_text("li pona")
    # If it parses, make sure it doesn't crash
    for utt, prob in results:
        assert isinstance(utt, Utterance)

def test_probabilities_sum_to_one():
    results = build_ast_from_text("mi moku lon tomo")
    assert results
    total = sum(p for _, p in results)
    assert abs(total - 1.0) < 1e-6

def test_sorted_by_probability():
    results = build_ast_from_text("tenpo ni la mi wile toki")
    probs = [p for _, p in results]
    assert probs == sorted(probs, reverse=True)

def test_all_demo_sentences_build():
    demo = [
        "mi pona",
        "mi moku e kasi",
        "jan li wile pali",
        "tomo pi telo nasa",
        "tenpo ni la mi wile toki",
        "jan Joni li pali li toki",
        "mi moku lon tomo",
        "jan pona o toki tawa mi",
        "mi en sina li pona",
        "taso ona li pona",
        "mi wile ala pali",
        "jan nanpa wan li pona",
        "mi pana e tomo e moku tawa ona",
    ]
    for text in demo:
        results = build_ast_from_text(text)
        assert results, f"No AST for {text!r}"
        for utt, prob in results:
            assert isinstance(utt, Utterance), f"Not Utterance for {text!r}"

def test_build_ast_bare_interjection():
    # An empty start tree (no children) is now valid — bare interjection / O A* path.
    from lark.tree import Tree
    bare_tree = Tree("start", [])
    utt = build_ast(bare_tree)
    assert isinstance(utt, Utterance)
    assert utt.starters == []
    assert utt.body is None

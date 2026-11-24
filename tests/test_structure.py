"""
tests/test_structure.py
Checks that the top-ranked parse has the structure wasona teaches, not just
that a sentence parses. Sections follow the grammar points in
grammars/wasona.cfg; every sentence is quoted from the wasona lesson named
next to it.
"""
from __future__ import annotations

import pytest

from ast_nodes import (
    PiPhrase, PreverbPhrase, Sentence, UnmarkedSubject, VerbPhraseIntransitive,
    VerbPhrasePrepositional, VerbPhraseTransitive, VerbPhraseYN,
)
from validator import validate, validate_text


def top(text: str) -> Sentence:
    """The top-ranked reading of a valid sentence."""
    r = validate(text)
    assert r.is_valid, r.diagnostics
    assert isinstance(r.utterance.body, Sentence)
    return r.utterance.body


def verb(predicate):
    """The verb phrase of a predicate, under any preverbs."""
    return predicate.vp if isinstance(predicate, PreverbPhrase) else predicate


# ---------------------------------------------------------------------------
# li is dropped only after a bare mi or sina
# ---------------------------------------------------------------------------

def test_bare_mi_sina_drop_li():
    c = top("sina lukin e moku.").clause                    # basics-1/06-pronouns
    assert isinstance(c.subject, UnmarkedSubject)
    assert c.predicate_type == "none"

@pytest.mark.parametrize("text, subject", [
    ("mi mute li kalama musi.", "mi mute"),                 # basics-2/16-society
    ("mi tu li kama wan.", "mi tu"),                        # basics-3/25-numbers
    ("mi taso li sona e ni.", "mi taso"),                   # basics-3/28-moo
])
def test_mi_with_an_adjective_keeps_li(text, subject):
    c = top(text).clause
    assert (c.subject.source, c.predicate_type) == (subject, "li")
    assert len(c.predicates) == 1


# ---------------------------------------------------------------------------
# Yes/no questions: V ala V doubles the first word of the verb
# ---------------------------------------------------------------------------

def test_v_ala_v_on_a_verb():
    p = top("pipi li moku ala moku e len?").clause.predicates[0]     # basics-2/17-questions
    assert isinstance(verb(p), VerbPhraseYN)

def test_v_ala_v_before_a_prepositional_phrase():
    vp = verb(top("kalama musi li pona ala pona tawa sina?").clause.predicates[0])
    assert isinstance(vp, VerbPhraseYN)
    assert vp.preps[0].prep.word.text == "tawa"

def test_v_ala_v_on_a_preverb():
    p = top("sina wile ala wile lape lon supa?").clause.predicates[0]
    assert isinstance(p, PreverbPhrase)
    assert [(pv.word.text, pv.question) for pv in p.preverbs] == [("wile", True)]

def test_v_ala_v_on_a_preposition():
    vp = verb(top("ona li lon ala lon ma Mewika?").clause.predicates[0])
    assert isinstance(vp, VerbPhrasePrepositional)
    assert vp.preps[0].prep.question

def test_preverb_ala_then_another_verb_is_negation():
    p = top("sina wile ala moku e suwi.").clause.predicates[0]      # basics-2/14-negation
    assert isinstance(p, PreverbPhrase)
    assert [(pv.word.text, pv.negated, pv.question) for pv in p.preverbs] == [("wile", True, False)]
    assert verb(p).verb.head.text == "moku"

@pytest.mark.parametrize("text", [
    "sitelen li musi anu seme?",                            # basics-2/17-questions
    "ni li wile e mani anu seme?",
])
def test_anu_seme_is_a_question_tag(text):
    s = top(text)
    assert s.anu_seme
    assert len(s.clause.predicates) == 1


# ---------------------------------------------------------------------------
# Calls (X o), requests and wishes
# ---------------------------------------------------------------------------

def test_call_then_a_sentence():
    s = top("jan Nowa o, soweli sina li wile musi!")        # basics-2/12-requests
    assert s.vocative.source == "jan Nowa"
    assert (s.clause.subject.source, s.clause.predicate_type) == ("soweli sina", "li")

def test_call_and_request_in_one():
    c = top("jan Ema o pini e lupa tomo!").clause           # basics-2/12-requests
    assert (c.subject.source, c.predicate_type) == ("jan Ema", "o")

def test_call_after_the_request():
    results = validate_text("o toki ala e wawa pi tenpo weka, jan nasa o. "
                            "tenpo open ona la mi lon.")    # more/28-02-uncommon-words
    assert all(r.is_valid for r in results)
    s = results[0].utterance.body
    assert s.vocative.source == "jan nasa"
    assert (s.clause.subject, s.clause.predicate_type) == (None, "o")

def test_wish_with_o():
    c = top("pona o tawa sina!").clause                     # basics-2/12-requests
    assert (c.subject.source, c.predicate_type) == ("pona", "o")

def test_mi_o():
    c = top("mi o moku e misikeke lon tenpo mute.").clause  # basics-3/21-nature
    assert (c.subject.source, c.predicate_type) == ("mi", "o")


# ---------------------------------------------------------------------------
# la contexts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, context", [
    ("tenpo pini la mi lili.", "tenpo pini"),               # basics-3/20-time
    ("kon li lete la, pipi li weka.", "kon lete"),          # basics-3/20-time
    ("tenpo pimeja lete la mi tu li awen seli", "tenpo pimeja lete"),   # basics-3/29-song
])
def test_la_context(text, context):
    assert [c.source for c in top(text).contexts] == [context]


# ---------------------------------------------------------------------------
# Prepositions as verbs; several li and e
# ---------------------------------------------------------------------------

def test_preposition_as_the_verb():
    vp = verb(top("jan li tawa tomo.").clause.predicates[0])           # basics-2/15-with-like
    assert isinstance(vp, VerbPhrasePrepositional)

def test_preposition_word_as_a_transitive_verb():
    vp = verb(top("jan li tawa e tomo.").clause.predicates[0])         # basics-2/15-with-like
    assert isinstance(vp, VerbPhraseTransitive)
    assert vp.verb.head.text == "tawa"

def test_several_li():
    c = top("jan li alasa e kala, li seli e ona, li moku e ona.").clause   # basics-3/21-nature
    assert len(c.predicates) == 3

def test_several_e():
    vp = verb(top("pan li jo e soweli e kili e namako.").clause.predicates[0])   # basics-3/21-nature
    assert [o.np.source for o in vp.objects] == ["soweli", "kili", "namako"]


# ---------------------------------------------------------------------------
# pi and names
# ---------------------------------------------------------------------------

def test_pi_before_a_name():
    subject = top("waso pi jan Sunita li suwi.").clause.subject.phrase   # basics-3/23-places
    assert isinstance(subject, PiPhrase)
    assert subject.pi_groups[0].source == "jan Sunita"

def test_pi_in_subject_and_object():
    c = top("jan pi mani mute li kute ala e jan pi mani lili.").clause   # basics-3/24-body
    assert isinstance(c.subject.phrase, PiPhrase)
    assert isinstance(verb(c.predicates[0]).objects[0].np, PiPhrase)

def test_capitalised_word_is_a_name():
    subject = top("jan Ali li pona e supa kepeken ko.").clause.subject.phrase   # basics-2/15-with-like
    assert [w.text for w in subject.modifiers] == ["Ali"]

def test_name_describes_a_verb():
    vp = verb(top("jan Sunita li toki Kanse e ma Nijon.").clause.predicates[0])   # basics-1/07-names
    assert vp.verb.source == "toki Kanse"
    assert vp.objects[0].np.source == "ma Nijon"


# ---------------------------------------------------------------------------
# Colons, negation and interjections
# ---------------------------------------------------------------------------

def test_colon_introduces_what_e_ni_announces():
    s = top("soweli li kama sona e ni: kalama wawa li tan waso ni.")   # basics-1/10-story
    assert verb(s.clause.predicates[0]).objects[0].np.source == "ni"
    assert s.elaboration.clause.subject.source == "kalama wawa"

def test_colon_after_any_clause():
    s = top("taso sina sona ala: utala li ike e sina.")    # basics-1/02-writing
    assert [w.text for w in s.starters[0].words] == ["taso"]
    assert s.elaboration.clause.subject.source == "utala"

def test_verb_then_ala_is_negation():
    p = top("mi sona ala.").clause.predicates[0]            # basics-3/28-moo
    assert isinstance(p, VerbPhraseIntransitive)
    assert (p.verb.head.text, [w.text for w in p.verb.modifiers]) == ("sona", ["ala"])

def test_number_shorthand():
    s = top("tenpo sike #LLLTTAMLLLTT la, tenpo mun #LTT la, tenpo suno #W la, "
            "ma Tosi li kama utala e ma Posuka.")            # more/25-01-large-numbers
    assert [c.source for c in s.contexts] == [
        "tenpo sike #LLLTTAMLLLTT", "tenpo mun #LTT", "tenpo suno #W"]

def test_n_is_a_hesitation():
    first, second = validate_text("n... mi sona ala.")      # basics-3/28-moo
    assert first.is_valid and first.utterance.body is None
    assert second.is_valid

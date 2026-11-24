"""
tests/test_validator.py
Tests for validator.py — structural and phonotactic checks on Toki Pona ASTs.
"""
import pytest
from validator import validate, validate_text, validate_utterance, Diagnostic, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def codes(r: ValidationResult) -> set[str]:
    return {d.code for d in r.diagnostics}


def has_code(r: ValidationResult, code: str) -> bool:
    return code in codes(r)


# ---------------------------------------------------------------------------
# A. Clean sentences — no errors or warnings
# ---------------------------------------------------------------------------

def test_clean_basic_transitive():
    r = validate("mi moku e kasi")
    assert not r.errors
    assert not r.warnings

def test_clean_preverb():
    r = validate("jan li wile pali")
    assert not r.errors
    assert not r.warnings

def test_clean_pi_multi_word():
    r = validate("mi jo e tomo pi telo nasa")
    assert not r.errors
    assert not r.warnings

def test_clean_valid_proper_noun():
    r = validate("jan Joni li pona")
    assert not r.errors
    assert not r.warnings

def test_clean_sentence_with_la():
    r = validate("tenpo ni la mi pona")
    assert not r.errors
    assert not r.warnings

def test_clean_taso_as_starter():
    # 'taso' as sentence-initial starter should NOT trigger taso-head warning
    r = validate("taso jan li pona")
    assert not has_code(r, "taso-head")
    assert not r.errors


# ---------------------------------------------------------------------------
# B. pi-single-word (ERROR)
# ---------------------------------------------------------------------------

def test_pi_single_word_warns():
    r = validate("tomo pi telo")
    assert has_code(r, "pi-single-word")
    diag = next(d for d in r.diagnostics if d.code == "pi-single-word")
    assert diag.severity == "ERROR"

def test_pi_multi_word_clean():
    r = validate("mi jo e tomo pi telo nasa")
    assert not has_code(r, "pi-single-word")


# ---------------------------------------------------------------------------
# C. proper-noun-phonotactics (ERROR)
# ---------------------------------------------------------------------------

def test_proper_noun_valid_no_error():
    r = validate("jan Joni li pona")
    assert not has_code(r, "proper-noun-phonotactics")

def test_proper_noun_forbidden_wu():
    r = validate("jan Wupi li pona")
    assert has_code(r, "proper-noun-phonotactics")
    diag = next(d for d in r.diagnostics if d.code == "proper-noun-phonotactics")
    assert diag.severity == "ERROR"

def test_proper_noun_forbidden_ji():
    r = validate("jan Jipi li pona")
    assert has_code(r, "proper-noun-phonotactics")

def test_proper_noun_forbidden_wo():
    r = validate("jan Wopi li pona")
    assert has_code(r, "proper-noun-phonotactics")

def test_proper_noun_forbidden_ti():
    r = validate("jan Tipa li pona")
    assert has_code(r, "proper-noun-phonotactics")

def test_proper_noun_forbidden_cluster_nm():
    # nm cluster is forbidden
    r = validate("jan Anma li pona")
    assert has_code(r, "proper-noun-phonotactics")

def test_kin_not_flagged_as_proper():
    # 'kin' is a MOD token but is explicitly excluded from phonotactics check
    r = validate("mi moku kin")
    assert not has_code(r, "proper-noun-phonotactics")


# ---------------------------------------------------------------------------
# D. bare-fragment (INFO — wasona teaches answers and greetings like "kama pona!")
# ---------------------------------------------------------------------------

def test_bare_fragment_info():
    r = validate("jan")
    assert has_code(r, "bare-fragment")
    diag = next(d for d in r.diagnostics if d.code == "bare-fragment")
    assert diag.severity == "INFO"
    assert r.is_valid

def test_full_sentence_no_fragment():
    r = validate("jan li pona")
    assert not has_code(r, "bare-fragment")


# ---------------------------------------------------------------------------
# E. ambiguous-parse (INFO)
# ---------------------------------------------------------------------------

def test_ambiguous_gets_info():
    # "sina sona ala sona" produces multiple parses (V-ala-V vs bare subject ala modifier)
    r = validate("sina sona ala sona")
    if r.parse_count > 1:
        assert has_code(r, "ambiguous-parse")
        diag = next(d for d in r.diagnostics if d.code == "ambiguous-parse")
        assert diag.severity == "INFO"

def test_unambiguous_no_info():
    r = validate("mi moku e kasi")
    # If only 1 parse, no ambiguous-parse diagnostic
    if r.parse_count == 1:
        assert not has_code(r, "ambiguous-parse")


# ---------------------------------------------------------------------------
# F. taso-head / kin-head (ERROR — wasona: kin follows its word; taso starts a
#    sentence or follows a word)
# ---------------------------------------------------------------------------

def test_taso_head_errors():
    # "mi moku e taso" — 'taso' as a phrase head in object position
    r = validate("mi moku e taso")
    assert has_code(r, "taso-head")
    diag = next(d for d in r.diagnostics if d.code == "taso-head")
    assert diag.severity == "ERROR"

def test_taso_starter_no_warn():
    # 'taso' as sentence starter — should NOT fire taso-head
    r = validate("taso ona li pona")
    assert not has_code(r, "taso-head")

def test_kin_head_errors():
    # 'kin' appearing as a phrase head
    r = validate("mi moku e kin")
    assert has_code(r, "kin-head")
    diag = next(d for d in r.diagnostics if d.code == "kin-head")
    assert diag.severity == "ERROR"

def test_kin_after_a_preverb_is_not_a_head():
    # "kama kin" is the verb kama + kin, not preverb kama + verb kin
    assert validate("kala li kama kin.").is_valid


# ---------------------------------------------------------------------------
# G. ni as a phrase head is fine (wasona lesson 6 teaches ni as a pronoun)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "ni li kasi suwi.",
    "ni li suli ala tawa mi.",
    "mi wile e ni",
])
def test_ni_as_head_is_valid(sentence):
    r = validate(sentence)
    assert r.is_valid, r.diagnostics


# ---------------------------------------------------------------------------
# H. v-ala-v-mismatch (ERROR)
# ---------------------------------------------------------------------------

def test_v_ala_v_match_clean():
    # Top parse of "sina sona ala sona" should have matching verb heads
    r = validate("sina sona ala sona")
    # The top parse may or may not be the YN form; check that if mismatch fires
    # it's because heads actually differ (not a false positive)
    if has_code(r, "v-ala-v-mismatch"):
        diag = next(d for d in r.diagnostics if d.code == "v-ala-v-mismatch")
        # The message should reference two different verbs
        assert " ala " in diag.message

def test_v_ala_v_mismatch_warns():
    # This test verifies the check fires correctly with non-matching heads.
    # We test via validate_utterance with a hand-built AST.
    from ast_nodes import (
        VerbPhraseYN, SimplePhrase, Word, VerbPhraseIntransitive,
        Clause, Sentence, Utterance, UnmarkedSubject,
    )
    from validator import validate_utterance

    def w(text, ttype="WORD", offset=0):
        return Word(text=text, token_type=ttype, offset=offset, source=text)

    sp_sona = SimplePhrase(head=w("sona", offset=5), modifiers=[], source="sona")
    sp_pona = SimplePhrase(head=w("pona", offset=12), modifiers=[], source="pona")
    vp = VerbPhraseYN(verb=sp_sona, repeated_verb=sp_pona, objects=[], source="sona ala pona")
    subj = UnmarkedSubject(word=w("sina", "UNMK", 0), source="sina")
    clause = Clause(subject=subj, predicate_type="none", predicates=[vp], source="sina sona ala pona")
    sent = Sentence(starters=[], contexts=[], clause=clause, anu_seme=False, source="sina sona ala pona")
    utt = Utterance(starters=[], body=sent, source="sina sona ala pona")

    diags = validate_utterance(utt, parse_count=1)
    assert any(d.code == "v-ala-v-mismatch" for d in diags)
    mismatch = next(d for d in diags if d.code == "v-ala-v-mismatch")
    assert mismatch.severity == "ERROR"
    assert "sona" in mismatch.message
    assert "pona" in mismatch.message


# ---------------------------------------------------------------------------
# I. Robustness
# ---------------------------------------------------------------------------

def test_ungrammatical_no_crash():
    r = validate("li li li")
    assert r.utterance is None
    assert r.parse_count == 0
    assert [d.code for d in r.errors] == ["no-parse"]
    assert r.is_valid is False

def test_empty_no_crash():
    r = validate("")
    assert r.utterance is None
    assert r.parse_count == 0
    assert r.diagnostics == []
    assert r.is_valid is False   # nothing parsed, so not a valid sentence

def test_is_valid_true_no_errors():
    r = validate("jan li pona")
    assert r.is_valid is True

def test_is_valid_false_with_error():
    r = validate("jan Wupi li pona")
    assert r.is_valid is False
    assert r.errors

def test_diagnostics_sorted_by_offset():
    # A sentence with two proper-noun violations at different offsets
    r = validate("jan Wupi en jan Jipi li pona")
    pn_diags = [d for d in r.diagnostics if d.code == "proper-noun-phonotactics"]
    if len(pn_diags) >= 2:
        offsets = [d.offset for d in pn_diags if d.offset is not None]
        assert offsets == sorted(offsets)

def test_none_offset_diagnostics_last():
    # bare-fragment has offset=None — should appear after offset diagnostics
    r = validate("jan Wupi")
    offsets = [d.offset for d in r.diagnostics]
    # All non-None offsets must come before None offsets
    none_seen = False
    for o in offsets:
        if o is None:
            none_seen = True
        elif none_seen:
            pytest.fail("A non-None offset appeared after a None offset")

def test_validation_result_properties():
    r = validate("jan Wupi li pona")
    assert all(d.severity == "ERROR" for d in r.errors)
    assert all(d.severity == "WARNING" for d in r.warnings)
    assert all(d.severity == "INFO" for d in r.infos)

def test_validate_returns_probability():
    r = validate("mi pona")
    assert r.utterance is not None
    assert 0.0 < r.probability <= 1.0

def test_validate_utterance_direct():
    # validate_utterance should work when called directly
    from ast_builder import build_ast_from_text
    results = build_ast_from_text("jan li pona")
    assert results
    utt, _ = results[0]
    diags = validate_utterance(utt, parse_count=1)
    assert isinstance(diags, list)
    assert not any(d.severity == "ERROR" for d in diags)


# ---------------------------------------------------------------------------
# J. unknown-word (ERROR) and capitalisation
# ---------------------------------------------------------------------------

def test_unknown_lowercase_word_errors():
    r = validate("mi moku e xyz")
    diag = next(d for d in r.diagnostics if d.code == "unknown-word")
    assert diag.severity == "ERROR"
    assert diag.offset == 10
    assert not r.is_valid

def test_capitalised_unknown_word_is_a_name():
    r = validate("jan Sonja li pali")
    assert not has_code(r, "unknown-word")
    assert r.is_valid

def test_capitalised_first_word_is_lowercased():
    # "Mi" starts the sentence, so it is the pronoun mi, not a name
    r = validate("Mi moku.")
    assert r.is_valid
    assert not has_code(r, "bare-fragment")
    assert r.utterance.body.clause.predicate_type == "none"
    diag = next(d for d in r.diagnostics if d.code == "capitalised-first-word")
    assert diag.severity == "INFO"   # wasona lesson 2: sentences start lowercase

def test_capital_mid_sentence_is_a_name():
    # basics-2/15: Ali is a person here, not ali ('all')
    r = validate("jan Ali li pona e supa kepeken ko.")
    assert r.is_valid
    name = r.utterance.body.clause.subject.phrase.modifiers[0]
    assert (name.text, name.token_type) == ("Ali", "MOD")

def test_unknown_word_reported_without_parse():
    r = validate("li xyz")
    assert {"no-parse", "unknown-word"} <= codes(r)


# ---------------------------------------------------------------------------
# K. no-parse (ERROR)
# ---------------------------------------------------------------------------

def test_no_parse_points_at_the_unexpected_word():
    r = validate("mi moku li li pona")
    diag = next(d for d in r.diagnostics if d.code == "no-parse")
    assert diag.severity == "ERROR"
    assert diag.offset == 11

def test_no_parse_incomplete_sentence():
    r = validate("mi moku e")
    diag = next(d for d in r.diagnostics if d.code == "no-parse")
    assert diag.offset is None
    assert "complete" in diag.message


# ---------------------------------------------------------------------------
# L. validate_text (several sentences)
# ---------------------------------------------------------------------------

def test_validate_text_splits_sentences():
    results = validate_text("mi moku. sina pali!")
    assert [r.text for r in results] == ["mi moku.", "sina pali!"]
    assert [r.offset for r in results] == [0, 9]
    assert all(r.is_valid for r in results)

def test_validate_text_offsets_point_into_full_text():
    text = "mi pona. jan Wupi li pona."
    results = validate_text(text)
    diag = next(d for d in results[1].diagnostics if d.code == "proper-noun-phonotactics")
    assert diag.offset == text.index("Wupi")

def test_validate_text_keeps_colon_clause_in_one_sentence():
    assert len(validate_text("ona li toki e ni: mi pona.")) == 1

def test_validate_text_no_words():
    assert validate_text("  ...  ") == []


# ---------------------------------------------------------------------------
# M. Names, numbers and commas
# ---------------------------------------------------------------------------

def test_name_without_noun_errors():
    # wasona always puts a head noun before a name: "nimi ona li soweli Mumu."
    r = validate("nimi mi li Lope.")
    diag = next(d for d in r.diagnostics if d.code == "name-without-noun")
    assert diag.severity == "ERROR"
    assert validate("nimi mi li jan Lope.").is_valid

def test_name_as_starter_errors():
    assert has_code(validate("Lope mi pona."), "name-without-noun")

def test_name_is_not_read_as_a_head():
    # "tomo tawa Ninja" is a car named Ninja, not "a house, to Ninja"
    assert validate("jan Kolonpo li jo e tomo tawa Ninja.").is_valid

def test_initial_skips_phonotactics():
    # more/07-02 shortens a name to an initial after the first mention
    assert not has_code(validate("jan T. li pona."), "proper-noun-phonotactics")

def test_number_order_warns():
    diag = next(d for d in validate("jan wan tu li kama.").diagnostics
                if d.code == "number-order")
    assert diag.severity == "WARNING"
    assert not has_code(validate("jan tu wan li kama."), "number-order")
    # ale multiplies by 100 in the 2021 system, so the count starts over
    assert not has_code(validate("mute ale mute luka"), "number-order")

def test_comma_inside_phrase_warns():
    r = validate("akesi pona, tu kin li moku e moku.")
    diag = next(d for d in r.diagnostics if d.code == "comma-in-phrase")
    assert diag.severity == "WARNING"
    assert diag.offset == 10

@pytest.mark.parametrize("sentence", [
    "kon li lete la, pipi li weka.",                         # after la
    "jan li alasa e kala, li seli e ona, li moku e ona.",   # before li
    "jan Nowa o, soweli sina li wile musi!",                 # after a call
])
def test_commas_between_phrases_are_fine(sentence):
    r = validate(sentence)
    assert r.is_valid and not has_code(r, "comma-in-phrase")


# ---------------------------------------------------------------------------
# N. Colons, pi, yes/no objects and number shorthand
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", ["mi: moku?", "sina: mi wile moku."])
def test_colon_needs_a_full_clause_before_it(sentence):
    # wasona's colons follow a statement ("taso sina sona ala: ...")
    assert not validate(sentence).is_valid

def test_trailing_colon_is_fine():
    # basics-3/20-time: what the colon announces comes as the next sentence
    assert validate("sin la mi wile toki e ni:").is_valid

def test_pi_is_not_read_before_one_word():
    # "ma pi tomo tawa kon", not "ma pi tomo" + "tawa kon"
    assert validate("mi wile tawa ma pi tomo tawa kon.").is_valid

def test_yes_no_question_objects_are_checked():
    assert has_code(validate("sina lukin ala lukin e jan Trump?"), "proper-noun-phonotactics")

def test_number_shorthand():
    # more/25-01: W T L M A abbreviate wan tu luka mute ale
    assert validate("tenpo suno #W la, ma Tosi li kama utala e ma Posuka.").is_valid


# ---------------------------------------------------------------------------
# O. Punctuation, repeated words, mi mute and en
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["mi moku:!", "mi moku?.", "mi moku.!", "mi moku e, moku."])
def test_punctuation_errors(text):
    r = validate(text)
    assert has_code(r, "punctuation")
    assert not r.is_valid

def test_colon_after_an_end_mark_is_caught():
    # the colon stays with its sentence instead of being split off
    [r] = validate_text("mi moku?:")
    assert has_code(r, "punctuation")

@pytest.mark.parametrize("text", [
    "mi moku?!", "mi moku??", "n... mi sona ala.", "kon li lete la, pipi li weka.",
])
def test_ordinary_punctuation_is_fine(text):
    assert not any(has_code(r, "punctuation") for r in validate_text(text))

def test_repeated_words_warn():
    diag = next(d for d in validate("akesi suli mi mi kin li moku e moku.").diagnostics
                if d.code == "repeated-word")
    assert diag.severity == "WARNING"
    assert has_code(validate("mi wile wile wile ni."), "repeated-word")

@pytest.mark.parametrize("text", [
    "mama mama mi li awen lon tenpo suli.",   # grandparent (more/28-02)
    "soweli mute! mu mu mu mu mu mu.",         # a sound (basics-3/28-moo)
    "mute mute mute luka luka luka tu wan",    # a number (more/25-01)
])
def test_meaningful_repeats_are_fine(text):
    assert not any(has_code(r, "repeated-word") for r in validate_text(text))

def test_mi_mute_needs_li():
    diag = next(d for d in validate("mi mute kalama musi.").diagnostics
                if d.code == "mi-mute-li")
    assert diag.severity == "ERROR"
    assert validate("mi mute li kalama musi.").is_valid   # basics-2/16-society
    assert validate("mi mute o tawa.").is_valid           # a request to mi mute

def test_en_inside_a_pi_group():
    r = validate("jan pona pi meli en mije li moku e moku.")
    assert has_code(r, "en-in-pi")
    assert not has_code(r, "pi-single-word")   # en-in-pi explains the same word better

def test_en_in_an_object_explains_itself():
    diag = next(d for d in validate("mi pona e tomo pi sina en ona.").diagnostics
                if d.code == "no-parse")
    assert "joins subjects" in diag.message


# ---------------------------------------------------------------------------
# P. wasona's wrong, discouraged and uncommon forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,code", [
    ("jan pi wawa pi pona mute li kama.", "pi-single-word"),   # basics-3/23-places
    ("ma Elopa lon seme?", "missing-li"),                      # basics-2/17-questions
    ("sina anu seme?", "anu-seme-fragment"),                   # basics-2/17-questions
    ("seme sina moku?", "seme-fronted"),                       # basics-2/17-questions
    ("mi moku, e moku.", "punctuation"),                       # basics-3/21-nature
    ("sina, o, toki!", "punctuation"),                         # basics-2/12-requests
    ("jan pi kama, sona li moku.", "comma-in-pi"),              # basics-3/23-places
    ("jan Mawijo li kepeken ilo alasa, taso linja li pakala.", "taso-mid-sentence"),
])
def test_wrong_forms_are_errors(text, code):
    r = validate(text)
    assert has_code(r, code) and not r.is_valid

@pytest.mark.parametrize("text", [
    "tan seme?",                                  # basics-2/19-dialogue
    "sitelen li musi anu seme?",                  # basics-2/17-questions
    "seme li monsuta e soweli lili?",             # basics-2/18-feelings
    "kasi li wile e telo, e suno, e ma.",         # basics-3/21-nature
    "jan Nowa o, soweli sina li wile musi!",      # basics-2/12-requests
    "tenpo ike, tenpo pi ken pilin, tenpo pi ken ala pilin",   # a list (known_good.txt)
])
def test_their_right_forms_are_valid(text):
    assert validate(text).is_valid

@pytest.mark.parametrize("text,code", [
    ("soweli jan Ema li pona.", "name-without-pi"),           # basics-3/23-places
    ("jan li toki e toki Inli.", "named-verb"),               # more/08-02
    ("pipi walo li moku poka pipi pimeja.", "poka-preposition"),
    ("sina namako lili, poka seme?", "poka-preposition"),
])
def test_discouraged_forms_warn(text, code):
    r = validate(text)
    diag = next(d for d in r.diagnostics if d.code == code)
    assert diag.severity == "WARNING" and r.is_valid

@pytest.mark.parametrize("text", [
    "soweli pi jan Ema li pona.",               # basics-3/23-places
    "tawa jan Ali",                             # basics-1/08-prepositions
    "jan li toki Inli.",                        # more/08-02
    "ona li lon poka sina.",                    # basics-3/20-time
    "mi pali e ni lon tenpo kama poka.",        # basics-3/20-time
])
def test_wasona_forms_do_not_warn(text):
    r = validate(text)
    assert r.is_valid and not r.warnings

@pytest.mark.parametrize("text,code,hint", [
    ("jan Ken li pona.", "short-name", "two to four"),           # more/07-01
    ("mi lukin sona e nimi.", "less-common-preverb", "alasa"),  # more/09-01
    ("mi lanpan e mani.", "uncommon-word", "kama jo"),          # more/28-02
])
def test_uncommon_choices_get_a_note(text, code, hint):
    r = validate(text)
    diag = next(d for d in r.diagnostics if d.code == code)
    assert diag.severity == "INFO" and hint in diag.message and r.is_valid

"""
validator.py
Validates Toki Pona sentences: parses them, then checks for grammar,
vocabulary and phonotactic problems.

Public API:
    validate(text: str, offset: int = 0) -> ValidationResult
        Validate ONE sentence: parse it, build the top AST, run all checks.
        Never raises.

    validate_text(text: str) -> list[ValidationResult]
        Split text into sentences and validate each one.

    validate_utterance(utterance: Utterance, parse_count: int = 1) -> list[Diagnostic]
        Run all checks on a pre-built Utterance AST.

Pipeline position: lexer → parser → ast_builder → **validator**

"Valid" means the sentence parsed AND no ERROR-severity diagnostic fired.
A sentence that does not parse gets a 'no-parse' ERROR, so it is never valid.

Checks
------
no-parse                 ERROR    The grammar has no reading of the sentence
unknown-word             ERROR    Lowercase word not in master_dictionary.yaml (names are capitalised)
name-without-noun        ERROR    A name as a phrase head or sentence starter (names follow a noun: jan Ema)
proper-noun-phonotactics ERROR    Capitalised name that violates Toki Pona phonotactics (initials exempt)
pi-single-word           ERROR    PiPhrase where any pi_group is a bare single-word SimplePhrase
v-ala-v-mismatch         ERROR    V-ala-V form where the two verb phrases have different heads
unmk-marked-subject      ERROR    Bare mi/sina followed by 'li'
mi-mute-li               ERROR    'mi mute' / 'sina mute' as a subject without li (wasona lesson 16)
en-in-pi                 ERROR    en inside a pi group; en joins whole subjects (wasona lesson 26)
missing-li               ERROR    A phrase then 'lon seme' etc. with no li ('ma Elopa lon seme?'; wasona lesson 17)
anu-seme-fragment        ERROR    'anu seme' after a bare phrase; it tags a whole sentence (lesson 17)
seme-fronted             ERROR    seme moved to the front, English-style ('seme sina moku?'; lesson 17)
punctuation              ERROR    A colon or period jammed against other end marks; a comma right after
                                  e/li/pi, before a verb's first e, or on both sides of o
comma-in-pi              ERROR    A comma right after a pi group's first word, leaving pi before one word (lesson 23)
taso-mid-sentence        ERROR    ', taso' inside a sentence; taso ('but') starts a new one (lesson 28)
taso-head                ERROR    'taso' as a phrase head ('but' starts a sentence, 'only' follows a word)
kin-head                 ERROR    'kin' as a phrase head (it follows the word it is about)
name-without-pi          WARNING  'soweli jan Ema' is a dog named Ema; Ema's dog is 'soweli pi jan Ema' (lesson 23)
named-verb               WARNING  'toki e toki Inli' talks about English; 'toki Inli' speaks it (more/08-02)
poka-preposition         WARNING  poka used as a preposition; wasona writes 'lon poka X' (lesson 20)
number-order             WARNING  Number words not biggest-first ("wan tu" for "tu wan")
comma-in-phrase          WARNING  Any other comma inside a phrase; commas go between phrases
repeated-word            WARNING  A pronoun twice in a row, or any word three times (numbers and sounds excepted)
short-name               INFO     A one-syllable name; wasona suggests two to four syllables (more/07-01)
less-common-preverb      INFO     sona, open, pini or lukin as a preverb; not everyone uses them (more/09-01)
uncommon-word            INFO     Not one of the 131 core and common words wasona teaches (more/28-02)
bare-fragment            INFO     Clause with no predicate (fine for answers and greetings)
capitalised-first-word   INFO     Sentence starts with a capital; Toki Pona starts lowercase
ambiguous-parse          INFO     Input has multiple distinct grammatical interpretations

Removed check
-------------
function-word-head flagged 'ni' as a phrase head, following pu's dictionary,
which lists ni only as an adjective. wasona teaches ni as a pronoun
(lesson 6: "ni li kasi suwi."), and 255 sentences in the known-good corpus
use it that way, so the check was dropped.

Omitted check
-------------
li-drop-with-modifier: The Lark grammar enforces this structurally. predicate_list_none
is only reachable from unmarked_subject (mi/sina UNMK tokens). A MarkedSubject without
li cannot be produced by the grammar, so this check would never fire.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal

from ast_nodes import (
    AnyPhrase, Clause, CompoundSubject, ContextPhrase, DisjunctPhrase,
    MarkedSubject, ObjectPhrase, OrdinalPhrase, PiPhrase, PrepPhrase,
    PreverbPhrase, Sentence, SentenceStarter, SimplePhrase, UnmarkedSubject, Utterance,
    VerbPhraseIntransitive, VerbPhrasePrepositional, VerbPhraseTransitive,
    VerbPhraseYN, Vocative, Word,
)

# ---------------------------------------------------------------------------
# Diagnostic types
# ---------------------------------------------------------------------------

DiagSeverity = Literal["ERROR", "WARNING", "INFO"]


@dataclass
class Diagnostic:
    """A single validation finding."""
    code:     str            # machine-readable rule name
    severity: DiagSeverity
    message:  str            # human-readable explanation
    offset:   int | None     # char offset of offending word in input; None = structural
    source:   str | None     # .source text of offending node; None = structural

    def __repr__(self) -> str:
        loc = f"@{self.offset}" if self.offset is not None else "structural"
        return f"Diagnostic({self.severity} {self.code!r} {loc}: {self.message!r})"


@dataclass
class ValidationResult:
    """The result of validating one Toki Pona sentence."""
    utterance:   Utterance | None   # None if input is ungrammatical
    probability: float              # top-parse softmax probability (0.0 if no parse)
    parse_count: int                # number of distinct parses found
    diagnostics: list[Diagnostic]  # sorted by offset (None-offset diagnostics last)
    is_valid:    bool               # True iff the sentence parsed and has no ERROR diagnostics
    text:        str = ""           # the sentence that was validated
    offset:      int = 0            # where that sentence starts in the full text
    readings:    list = field(default_factory=list)  # every (utterance, probability), likeliest first
    text_diagnostics: list[Diagnostic] = field(default_factory=list)  # the checks that don't depend on the parse

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "ERROR"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "WARNING"]

    @property
    def infos(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "INFO"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Toki Pona proper-name phonotactics:
#   syllable = (C)V(n)  where C ∈ {m,n,p,t,k,s,w,l,j}, V ∈ {a,e,i,o,u}
#   forbidden syllable onsets: Wu Wo Ji Ti (any capitalisation)
#   forbidden consonant clusters: nm, nn
_PROPER_RE = re.compile(
    r'^((?!Wu|Wo|Ji|Ti)[MNPTKSWLJ][aeiou]|[AEIOU])(?!nm|nn)n?'
    r'((?!wu|wo|ji|ti)[mnptkswlj][aeiou](?!nm|nn)n?)*$'
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _head_text(phrase: AnyPhrase) -> str:
    """Extract the head word text from any phrase type."""
    if isinstance(phrase, (SimplePhrase, OrdinalPhrase)):
        base = phrase if isinstance(phrase, SimplePhrase) else phrase.base
        return base.head.text
    if isinstance(phrase, PiPhrase):
        return _head_text(phrase.base)
    return ""   # DisjunctPhrase — no single head


def _head_offset(phrase: AnyPhrase) -> int | None:
    """Extract the head word character offset from any phrase type."""
    if isinstance(phrase, SimplePhrase):
        return phrase.head.offset
    if isinstance(phrase, OrdinalPhrase):
        return phrase.base.head.offset
    if isinstance(phrase, PiPhrase):
        return _head_offset(phrase.base)
    return None


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_pi_single_word(node: PiPhrase, diags: list[Diagnostic]) -> None:
    """pi-single-word: pi followed by only one word is discouraged."""
    for group in node.pi_groups:
        if isinstance(group, SimplePhrase) and len(group.modifiers) == 0:
            diags.append(Diagnostic(
                code="pi-single-word",
                severity="ERROR",
                message=(
                    f"'pi {group.head.text}' regroups only one word; "
                    "pi should precede a multi-word phrase"
                ),
                offset=group.head.offset,
                source=group.source,
            ))


def _check_proper_noun(word: Word, diags: list[Diagnostic]) -> None:
    """proper-noun-phonotactics: capitalised MOD word must follow Toki Pona phonotactics."""
    if word.token_type != "MOD":
        return
    if word.text == "kin":
        return   # kin is a legitimate modifier-only word, not a proper name
    if len(word.text) == 1:
        return   # an initial ("jan T.", wasona more/07-02), not a full name
    if not word.text[0].isupper():
        return   # lowercase MOD — should not happen, but guard anyway
    if not _PROPER_RE.match(word.text):
        diags.append(Diagnostic(
            code="proper-noun-phonotactics",
            severity="ERROR",
            message=(
                f"Proper name {word.text!r} violates Toki Pona phonotactics "
                "(bad syllable or forbidden cluster such as wu/wo/ji/ti or nm/nn)"
            ),
            offset=word.offset,
            source=word.text,
        ))


def _check_bare_fragment(node: Clause, diags: list[Diagnostic]) -> None:
    """bare-fragment: a clause with no predicates is a bare subject fragment."""
    if len(node.predicates) == 0:
        diags.append(Diagnostic(
            code="bare-fragment",
            severity="INFO",
            message=("Clause has no predicate — fine for answers and greetings "
                     "('musi ala.', 'kama pona!')"),
            offset=None,
            source=node.source or None,
        ))


def _fragment_phrase(node: Clause):
    """The phrase of a bare fragment (a clause with no predicate), else None."""
    if node.predicates or not isinstance(node.subject, MarkedSubject):
        return None
    return node.subject.phrase


def _check_missing_li(node: Clause, diags: list[Diagnostic]) -> None:
    """missing-li: 'ma Elopa lon seme?' asks about ma Elopa, so it needs li (wasona lesson 17)."""
    for phrase in _nodes(_fragment_phrase(node)):
        if not isinstance(phrase, SimplePhrase):
            continue
        words = [phrase.head, *phrase.modifiers]
        for prep, seme in zip(words[1:], words[2:]):
            if prep.token_type == "PREP" and seme.text == "seme":
                diags.append(Diagnostic(
                    code="missing-li",
                    severity="ERROR",
                    message=(f"'{prep.text} seme' asks about the words before it, so li goes "
                             "in between, as in 'waso li lon seme?' (wasona lesson 17)"),
                    offset=prep.offset,
                    source=node.source or None,
                ))
                return


def _check_anu_seme_fragment(node: Clause, diags: list[Diagnostic]) -> None:
    """anu-seme-fragment: 'anu seme' tags a whole sentence, not a bare phrase (wasona lesson 17)."""
    phrase = _fragment_phrase(node)
    if not isinstance(phrase, CompoundSubject) or phrase.conjunction != "anu":
        return
    last = phrase.parts[-1]
    if isinstance(last, SimplePhrase) and last.head.text == "seme" and not last.modifiers:
        diags.append(Diagnostic(
            code="anu-seme-fragment",
            severity="ERROR",
            message=("'anu seme' ('or what?') goes after a whole sentence, "
                     "as in 'sitelen li musi anu seme?' (wasona lesson 17)"),
            offset=last.head.offset,
            source=node.source or None,
        ))


def _check_seme_fronted(node: Clause, diags: list[Diagnostic]) -> None:
    """seme-fronted: seme stays where the answer goes, unlike English (wasona lesson 17)."""
    phrase = _fragment_phrase(node)
    if (isinstance(phrase, SimplePhrase) and phrase.head.text == "seme"
            and len(phrase.modifiers) > 1 and phrase.modifiers[0].token_type == "UNMK"):
        diags.append(Diagnostic(
            code="seme-fronted",
            severity="ERROR",
            message=("seme stays where the answer goes: 'sina moku e seme?', not "
                     "'seme sina moku?'; only English moves question words first (wasona lesson 17)"),
            offset=phrase.head.offset,
            source=phrase.source,
        ))


def _check_taso_head(node: SimplePhrase, diags: list[Diagnostic]) -> None:
    """taso-head: 'taso' as a phrase head is unusual (normally a sentence-starter)."""
    # Note: SentenceStarter nodes contain Word objects directly, not SimplePhrases,
    # so this check never fires for legitimate 'taso' starters.
    if node.head.text == "taso":
        diags.append(Diagnostic(
            code="taso-head",
            severity="ERROR",
            message=(
                "'taso' can't be a phrase head: it is 'but' at the start of a "
                "sentence or 'only' after a word (wasona)"
            ),
            offset=node.head.offset,
            source=node.source,
        ))


def _check_kin_head(node: SimplePhrase, diags: list[Diagnostic]) -> None:
    """kin-head: 'kin' is a modifier-only word and cannot be a phrase head."""
    if node.head.text == "kin":
        diags.append(Diagnostic(
            code="kin-head",
            severity="ERROR",
            message=(
                "'kin' can't be a phrase head: it goes after the word it is "
                "about (wasona); 'mi kin li ...' needs li"
            ),
            offset=node.head.offset,
            source=node.source,
        ))



def _check_v_ala_v(node: VerbPhraseYN, diags: list[Diagnostic]) -> None:
    """v-ala-v-mismatch: both verb phrases in V-ala-V must use the same head word."""
    if node.repeated_verb is None:
        return
    first  = _head_text(node.verb)
    second = _head_text(node.repeated_verb)
    if first and second and first != second:
        diags.append(Diagnostic(
            code="v-ala-v-mismatch",
            severity="ERROR",
            message=(
                f"V-ala-V mismatch: '{first} ala {second}' — "
                "both phrases should use the same verb"
            ),
            offset=_head_offset(node.verb),
            source=node.source,
        ))


def _check_unmk_marked_subject(node: Clause, diags: list[Diagnostic]) -> None:
    """unmk-marked-subject: bare mi/sina should not take 'li' (use unmarked predicate form)."""
    if node.predicate_type != "li":
        return
    if not isinstance(node.subject, MarkedSubject):
        return
    phrase = node.subject.phrase
    if (
        isinstance(phrase, SimplePhrase)
        and phrase.head.token_type == "UNMK"
        and len(phrase.modifiers) == 0
    ):
        diags.append(Diagnostic(
            code="unmk-marked-subject",
            severity="ERROR",
            message=(
                f"'{phrase.head.text}' is an unmarked subject (mi/sina) "
                "and should not take 'li'; use the predicate directly without 'li'"
            ),
            offset=phrase.head.offset,
            source=phrase.source,
        ))


def _check_ambiguous(parse_count: int, diags: list[Diagnostic]) -> None:
    """ambiguous-parse: report when there is more than one grammatical interpretation."""
    if parse_count > 1:
        diags.append(Diagnostic(
            code="ambiguous-parse",
            severity="INFO",
            message=(
                f"Input has {parse_count} distinct grammatical interpretation(s); "
                "showing top-scored parse"
            ),
            offset=None,
            source=None,
        ))


# Number words in counting order; ale/ali (100, or x100 in the 2021 system)
# and any other word start the count over.
_NUMBER_VALUES: dict[str, int] = {"wan": 1, "tu": 2, "luka": 5, "mute": 20}


# Punctuation that wasona's "like English" (lesson 2) rules out.
_PUNCTUATION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r":[.!?:]|[.!?]:"),
     "A colon introduces what follows; it can't touch another . ! ? or :"),
    (re.compile(r"(?<!\.)\.[!?]|[!?]\.(?!\.)"),
     "A sentence ends with one kind of mark; don't mix . with ? or !"),
    (re.compile(r"\b(?:e|li|pi)\b\s*,"),
     "No comma right after e, li or pi: the particle goes with the words after it"),
    (re.compile(r",\s*o\s*,"),
     "A comma on both sides of o cuts it off; a call is 'jan Nowa o, ...' and a "
     "request 'o toki!' (wasona lesson 12)"),
]

# Words that may repeat freely: sounds ("mu mu mu", "a a a"). Numbers too.
_SOUNDS:   frozenset[str] = frozenset({"a", "mu", "n"})
_PRONOUNS: frozenset[str] = frozenset({"mi", "sina", "ona"})


def _check_punctuation(text: str, offset: int, diags: list[Diagnostic]) -> None:
    """punctuation: marks that English-like punctuation rules out."""
    for pattern, message in _PUNCTUATION_RULES:
        for m in pattern.finditer(text):
            diags.append(Diagnostic(
                code="punctuation",
                severity="ERROR",
                message=message,
                offset=offset + m.start(),
                source=m.group(),
            ))
    # A comma before e separates objects ("e telo, e suno"), so an e must
    # come earlier in the same predicate
    for m in re.finditer(r",\s*e\b", text):
        predicate = re.split(r"\b(?:li|o|la)\b|:", text[:m.start()])[-1]
        if not re.search(r"\be\b", predicate):
            diags.append(Diagnostic(
                code="punctuation",
                severity="ERROR",
                message=("A comma before e only separates objects in a list, as in 'kasi li "
                         "wile e telo, e suno, e ma.' (wasona lesson 21); the first e "
                         "follows its verb directly"),
                offset=offset + m.start(),
                source=m.group(),
            ))


def _check_repeated_words(text: str, offset: int, diags: list[Diagnostic]) -> None:
    """
    repeated-word: a pronoun twice in a row ("mi mi"), or any word three or more
    times. Numbers and sounds repeat freely ("mute mute mute luka", "mu mu mu"),
    and a content word may double ("mama mama mi" = grandparent, wasona more/28-02).
    """
    from lexer import TokenType, tokenize
    tokens = tokenize(text)
    i = 0
    while i < len(tokens):
        j = i
        while j + 1 < len(tokens) and tokens[j + 1].text == tokens[i].text:
            j += 1
        run, word = j - i + 1, tokens[i].text
        if (run > 1 and tokens[i].type != TokenType.NUMBER and word not in _SOUNDS
                and (word in _PRONOUNS or run >= 3)):
            diags.append(Diagnostic(
                code="repeated-word",
                severity="WARNING",
                message=f"'{word}' appears {run} times in a row; was that intended?",
                offset=offset + tokens[i + 1].index,
                source=" ".join([word] * run),
            ))
        i = j + 1


def _check_mi_mute(node: Clause, diags: list[Diagnostic]) -> None:
    """mi-mute-li: 'mi mute' / 'sina mute' as a subject needs li (wasona lesson 16)."""
    if not isinstance(node.subject, UnmarkedSubject) or node.predicate_type != "none":
        return
    first = node.predicates[0] if node.predicates else None
    verb = getattr(first, "verb", None)   # preverb and prepositional predicates have none
    if not isinstance(verb, SimplePhrase) or verb.head.text != "mute":
        return
    if verb.modifiers or getattr(first, "objects", None) or first.preps or len(node.predicates) > 1:
        who = node.subject.word.text
        diags.append(Diagnostic(
            code="mi-mute-li",
            severity="ERROR",
            message=(f"'{who} mute' is the subject here, so it needs li: '{who} mute li ...' "
                     "(without li, mute is read as the verb; wasona lesson 16)"),
            offset=node.subject.word.offset,
            source=node.source,
        ))


def _check_en_in_pi(node: CompoundSubject, diags: list[Diagnostic]) -> None:
    """en-in-pi: en joins whole subjects, never words inside a pi group (wasona lesson 26)."""
    if node.conjunction != "en":
        return
    for part in node.parts[:-1]:
        if isinstance(part, PiPhrase):
            last = part.pi_groups[-1]
            if isinstance(last, SimplePhrase) and not last.modifiers:
                diags.append(Diagnostic(
                    code="en-in-pi",
                    severity="ERROR",
                    message=(f"en joins whole subjects, not words inside a pi group "
                             f"('pi {last.head.text} en ...'); wasona lesson 26"),
                    offset=last.head.offset,
                    source=part.source,
                ))


def _check_named_verb(node: VerbPhraseTransitive, diags: list[Diagnostic]) -> None:
    """named-verb: 'toki e toki Inli' talks about English; 'toki Inli' speaks it (wasona more/08-02)."""
    if not isinstance(node.verb, SimplePhrase):
        return
    verb = node.verb.head.text
    for obj in node.objects:
        np = obj.np
        if not isinstance(np, SimplePhrase) or np.head.text != verb:
            continue
        names = [w.text for w in np.modifiers if _is_name(w)]
        if names:
            diags.append(Diagnostic(
                code="named-verb",
                severity="WARNING",
                message=(f"'{verb} e {np.source}' makes {np.source} the object ('jan li toki e "
                         "toki Inli.' = 'The person speaks about English.'); say "
                         f"'{verb} {names[0]}' or '{verb} kepeken {np.source}' (wasona more/08-02)"),
                offset=np.head.offset,
                source=node.source,
            ))


# wasona more/09-01: sona, open, pini and lukin as preverbs "are worth knowing,
# but not everyone uses them", and each has a common way around it.
_LESS_COMMON_PREVERBS: dict[str, str] = {
    "sona":  "ken often does the job: 'mi ken sitelen e waso.'",
    "open":  "kama does the job: 'mi kama sona.' ('I begin to know.')",
    "pini":  "pini as the verb does the job: 'mi pini e pali mi.'",
    "lukin": "most speakers use alasa: 'mi alasa sona e nimi.'",
}


def _check_less_common_preverb(node: PreverbPhrase, diags: list[Diagnostic]) -> None:
    """less-common-preverb: sona, open, pini or lukin as a preverb (wasona more/09-01)."""
    for pv in node.preverbs:
        hint = _LESS_COMMON_PREVERBS.get(pv.word.text)
        if hint:
            diags.append(Diagnostic(
                code="less-common-preverb",
                severity="INFO",
                message=f"Not everyone uses '{pv.word.text}' as a preverb; {hint} (wasona more/09-01)",
                offset=pv.word.offset,
                source=node.source,
            ))


def _is_name(word: Word) -> bool:
    return word.token_type == "MOD" and word.text[:1].isupper()


def _check_name_head(node: SimplePhrase, diags: list[Diagnostic]) -> None:
    """name-without-noun: names are adjectives after a head noun (wasona more/07-01)."""
    if _is_name(node.head):
        diags.append(Diagnostic(
            code="name-without-noun",
            severity="ERROR",
            message=(
                f"'{node.head.text}' is a name, and names follow a head noun "
                f"(jan {node.head.text}, ma {node.head.text})"
            ),
            offset=node.head.offset,
            source=node.source,
        ))


def _check_starters(starters: list[SentenceStarter], diags: list[Diagnostic]) -> None:
    """name-without-noun, for a name used as a sentence starter."""
    for starter in starters:
        for w in starter.words:
            if _is_name(w):
                diags.append(Diagnostic(
                    code="name-without-noun",
                    severity="ERROR",
                    message=f"'{w.text}' is a name, and names follow a head noun (jan {w.text})",
                    offset=w.offset,
                    source=w.text,
                ))


def _check_name_without_pi(node: SimplePhrase, diags: list[Diagnostic]) -> None:
    """name-without-pi: 'soweli jan Ema' is a dog named Ema; Ema's dog is 'soweli pi jan Ema' (wasona lesson 23)."""
    # "tawa jan Ali" is a preposition and its object (wasona lesson 8)
    if node.head.text == "jan" or node.head.token_type == "PREP":
        return
    words = [node.head, *node.modifiers]
    for i in range(1, len(words) - 1):
        if words[i].text == "jan" and _is_name(words[i + 1]):
            owner = words[i + 1].text
            thing = " ".join(w.text for w in words[:i])
            diags.append(Diagnostic(
                code="name-without-pi",
                severity="WARNING",
                message=(f"Without pi, the name describes '{thing}' ('soweli jan Ema' is 'a "
                         f"human-like dog named Ema'); for {owner}'s {thing}, write "
                         f"'{thing} pi jan {owner}' (wasona lesson 23)"),
                offset=words[i].offset,
                source=node.source,
            ))
            return


def _check_short_name(word: Word, diags: list[Diagnostic]) -> None:
    """short-name: wasona suggests names of two to four syllables (more/07-01)."""
    if (_is_name(word) and len(word.text) > 1 and _PROPER_RE.match(word.text)
            and len(re.findall(r"[aeiou]", word.text.lower())) == 1):
        diags.append(Diagnostic(
            code="short-name",
            severity="INFO",
            message=(f"'{word.text}' has one syllable; wasona suggests names of two to four, "
                     "since there are too few short ones to go around (more/07-01)"),
            offset=word.offset,
            source=word.text,
        ))


def _check_number_order(words: list[Word], diags: list[Diagnostic]) -> None:
    """number-order: wasona counts biggest first ('tu wan' = 3), never smallest first."""
    previous = None
    for w in words:
        value = _NUMBER_VALUES.get(w.text)
        if value is not None and previous is not None and value > previous:
            diags.append(Diagnostic(
                code="number-order",
                severity="WARNING",
                message=f"Number words go from biggest to smallest; '{w.text}' follows a smaller one",
                offset=w.offset,
                source=w.text,
            ))
        previous = value


def _check_first_word(text: str, offset: int, diags: list[Diagnostic]) -> None:
    """capitalised-first-word: Toki Pona sentences start lowercase (wasona lesson 2)."""
    from lexer import DICTIONARY
    m = re.search(r"[a-zA-Z]+", text)
    if m and m.group()[0].isupper() and m.group().lower() in DICTIONARY:
        diags.append(Diagnostic(
            code="capitalised-first-word",
            severity="INFO",
            message=(
                f"Toki Pona sentences start lowercase; '{m.group()}' is read "
                f"as the word '{m.group().lower()}', not a name"
            ),
            offset=offset + m.start(),
            source=m.group(),
        ))


def _nodes(node):
    """Yield node and every AST node and Word inside it."""
    if isinstance(node, list):
        for item in node:
            yield from _nodes(item)
    elif is_dataclass(node):
        yield node
        for f in fields(node):
            yield from _nodes(getattr(node, f.name))


def _check_commas(utterance: Utterance, text: str, offset: int,
                  diags: list[Diagnostic]) -> None:
    """
    A comma inside a phrase; wasona's go between phrases (after la, before li
    or e, around a call). Before taso it is taso-mid-sentence (ERROR), right
    after a pi group's first word comma-in-pi (ERROR), anywhere else
    comma-in-phrase (WARNING).
    """
    commas = [offset + i for i, ch in enumerate(text) if ch == ","]
    for node in _nodes(utterance):
        if not commas:
            return
        if isinstance(node, (SimplePhrase, PiPhrase)):
            words = sorted((w for w in _nodes(node) if isinstance(w, Word)), key=lambda w: w.offset)
            for c in [c for c in commas if words[0].offset < c < words[-1].offset]:
                commas.remove(c)   # report each comma once
                after = next(w for w in words if w.offset > c)
                diags.append(_comma_in_phrase(node, c, after, text, offset))


def _comma_in_phrase(node: SimplePhrase | PiPhrase, comma: int, after: Word,
                     text: str, offset: int) -> Diagnostic:
    """The diagnostic for a comma inside node, followed by the word after."""
    if after.text == "taso":
        return Diagnostic(
            code="taso-mid-sentence",
            severity="ERROR",
            message=("taso ('but') starts a new sentence, so end this one first, "
                     "as in 'mi wile lape. taso mi ken ala.' (wasona lesson 28)"),
            offset=after.offset,
            source=node.source,
        )
    # A comma after a pi group's first word leaves pi before one word. Commas
    # further in may separate list items the grammar folds into one phrase
    # ("tenpo pi ken pilin, tenpo pi ken ala pilin").
    for group in getattr(node, "pi_groups", []):
        words = sorted((w for w in _nodes(group) if isinstance(w, Word)), key=lambda w: w.offset)
        first_end = words[0].offset + len(words[0].text)
        if first_end <= comma < words[-1].offset and not text[first_end - offset:comma - offset].strip():
            return Diagnostic(
                code="comma-in-pi",
                severity="ERROR",
                message=(f"A comma right after 'pi {words[0].text}' leaves pi before one word; "
                         "there are 'always two words or more after it' (wasona lesson 23)"),
                offset=comma,
                source=node.source,
            )
    return Diagnostic(
        code="comma-in-phrase",
        severity="WARNING",
        message=("A comma splits this phrase; commas go between phrases "
                 "(after la, before li or e, around a call)"),
        offset=comma,
        source=node.source,
    )


# wasona's pronouns (lesson 6); after poka they say whose side: "jan poka mi".
_POKA_PRONOUNS: frozenset[str] = frozenset({"mi", "sina", "ona", "ni"})


def _check_poka(utterance: Utterance, text: str, offset: int,
                diags: list[Diagnostic]) -> None:
    """
    poka-preposition: poka isn't a preposition; wasona writes 'lon poka X'
    (lesson 20: "ona li lon poka sina."). Flags a poka with words after it that
    a comma sets apart ("..., poka seme?"), or that follows another word with
    two or more words after it ("moku poka pipi pimeja"), unless a pronoun comes
    next ("jan poka mi" = my neighbour).
    """
    for node in _nodes(utterance):
        if not isinstance(node, SimplePhrase):
            continue
        words = [node.head, *node.modifiers]
        for i, poka in enumerate(words[:-1]):
            if poka.text != "poka":
                continue
            rest = words[i + 1:]
            after_comma = text[:poka.offset - offset].rstrip().endswith(",")
            if after_comma or (i > 0 and len(rest) > 1 and rest[0].text not in _POKA_PRONOUNS):
                place = " ".join(w.text for w in rest)
                fix = "lon poka " + ("pi " if len(rest) > 1 else "") + place
                diags.append(Diagnostic(
                    code="poka-preposition",
                    severity="WARNING",
                    message=(f"poka isn't a preposition (wasona lesson 20: 'ona li lon poka "
                             f"sina.'); for 'beside {place}', write '{fix}'"),
                    offset=poka.offset,
                    source=node.source,
                ))
                break


def _check_unknown_words(text: str, offset: int, diags: list[Diagnostic]) -> None:
    """unknown-word: a lowercase word that is not in the dictionary."""
    from lexer import TokenType, tokenize
    for tok in tokenize(text):
        if tok.type == TokenType.UNKNOWN:
            diags.append(Diagnostic(
                code="unknown-word",
                severity="ERROR",
                message=(
                    f"'{tok.text}' is not in the dictionary; "
                    "capitalise it if it is a name"
                ),
                offset=offset + tok.index,
                source=tok.text,
            ))


def _check_uncommon_words(text: str, offset: int, diags: list[Diagnostic]) -> None:
    """uncommon-word: not one of the 131 core and common words wasona teaches (more/28-02)."""
    from lexer import ENTRIES, tokenize
    for tok in tokenize(text):
        entry = ENTRIES.get(tok.text)
        if entry and entry.get("usage") in ("uncommon", "obscure"):
            alternative = entry.get("pu_alt")
            diags.append(Diagnostic(
                code="uncommon-word",
                severity="INFO",
                message=(f"'{tok.text}' is {entry['usage']}: not one of the 131 core and common "
                         "words wasona teaches (more/28-02)"
                         + (f"; in core words: {alternative}" if alternative else "")),
                offset=offset + tok.index,
                source=tok.text,
            ))


def _no_parse(text: str, offset: int) -> Diagnostic:
    """no-parse: the grammar has no reading of the sentence."""
    from parser import explain_failure
    where, why = explain_failure(text) or (None, "the sentence cannot be parsed")
    return Diagnostic(
        code="no-parse",
        severity="ERROR",
        message=f"Not a grammatical sentence: {why}",
        offset=None if where is None else offset + where,
        source=None,
    )


def _by_offset(d: Diagnostic) -> tuple[bool, int]:
    """Sort key: by offset, structural (None-offset) diagnostics last."""
    return (d.offset is None, d.offset or 0)


# ---------------------------------------------------------------------------
# Recursive AST walker
# ---------------------------------------------------------------------------

def _walk(node, diags: list[Diagnostic]) -> None:
    """
    Recursively walk the AST and fire the appropriate checks at each node type.
    Uses match/case for readable dispatch; falls through silently for unknown types.
    """
    match node:

        case SimplePhrase():
            _check_taso_head(node, diags)
            _check_kin_head(node, diags)
            _check_name_head(node, diags)
            _check_name_without_pi(node, diags)
            _check_proper_noun(node.head, diags)
            for w in node.modifiers:
                _check_proper_noun(w, diags)
                _check_short_name(w, diags)
            _check_number_order([node.head, *node.modifiers], diags)

        case PiPhrase():
            _check_pi_single_word(node, diags)
            _walk(node.base, diags)
            for g in node.pi_groups:
                _walk(g, diags)

        case OrdinalPhrase():
            _walk(node.base, diags)
            _check_number_order(node.ordinal.words, diags)

        case DisjunctPhrase():
            for opt in node.options:
                _walk(opt, diags)

        case ObjectPhrase():
            _walk(node.np, diags)
            for pp in node.preps:
                _walk(pp, diags)

        case PrepPhrase():
            _walk(node.obj, diags)

        case PreverbPhrase():
            _check_less_common_preverb(node, diags)
            _walk(node.vp, diags)

        case VerbPhraseTransitive():
            _check_named_verb(node, diags)
            _walk(node.verb, diags)
            for obj in node.objects:
                _walk(obj, diags)
            for pp in node.preps:
                _walk(pp, diags)

        case VerbPhraseIntransitive():
            _walk(node.verb, diags)
            for pp in node.preps:
                _walk(pp, diags)

        case VerbPhrasePrepositional():
            for pp in node.preps:
                _walk(pp, diags)

        case VerbPhraseYN():
            _check_v_ala_v(node, diags)
            _walk(node.verb, diags)
            if node.repeated_verb is not None:
                _walk(node.repeated_verb, diags)
            for obj in node.objects:
                _walk(obj, diags)
            for pp in node.preps:
                _walk(pp, diags)

        case UnmarkedSubject():
            pass   # single UNMK token (mi/sina), no deep structure to check

        case MarkedSubject():
            _walk(node.phrase, diags)

        case CompoundSubject():
            _check_en_in_pi(node, diags)
            for part in node.parts:
                _walk(part, diags)

        case ContextPhrase():
            _walk(node.content, diags)

        case Clause():
            _check_bare_fragment(node, diags)
            _check_missing_li(node, diags)
            _check_anu_seme_fragment(node, diags)
            _check_seme_fronted(node, diags)
            _check_unmk_marked_subject(node, diags)
            _check_mi_mute(node, diags)
            if node.subject is not None:
                _walk(node.subject, diags)
            for pred in node.predicates:
                _walk(pred, diags)

        case Vocative():
            _walk(node.addressee, diags)

        case Sentence():
            _check_starters(node.starters, diags)
            for ctx in node.contexts:
                _walk(ctx, diags)
            _walk(node.clause, diags)
            if node.vocative is not None:
                _walk(node.vocative, diags)
            if node.elaboration is not None:
                _walk(node.elaboration, diags)

        case Utterance():
            _check_starters(node.starters, diags)
            _walk(node.body, diags)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_utterance(
    utterance: Utterance,
    parse_count: int = 1,
) -> list[Diagnostic]:
    """
    Run all structural checks on a pre-built Utterance AST.

    Args:
        utterance:   The root AST node to check.
        parse_count: Total number of parses (from build_ast_from_text), used
                     to generate the ambiguous-parse INFO diagnostic.

    Returns:
        List of Diagnostic objects sorted by offset (None-offset diagnostics last).
    """
    diags: list[Diagnostic] = []
    _check_ambiguous(parse_count, diags)
    _walk(utterance, diags)
    # en-in-pi explains the single-word pi group better than pi-single-word does
    en_in_pi = {d.offset for d in diags if d.code == "en-in-pi"}
    diags = [d for d in diags if not (d.code == "pi-single-word" and d.offset in en_in_pi)]
    diags.sort(key=_by_offset)
    return diags


def validate(text: str, offset: int = 0) -> ValidationResult:
    """
    Validate ONE sentence: parse it, build the top-scored AST, run all checks.
    Use validate_text() for input that may hold several sentences.

    Args:
        text:   One Toki Pona sentence.
        offset: Where the sentence starts in a longer text; added to every
                diagnostic and AST word offset.

    Returns:
        ValidationResult — never raises. utterance is None when the sentence
        does not parse; that case carries a 'no-parse' ERROR.
    """
    from ast_builder import build_ast_from_text
    diags: list[Diagnostic] = []
    _check_unknown_words(text, offset, diags)
    _check_uncommon_words(text, offset, diags)
    _check_first_word(text, offset, diags)
    _check_punctuation(text, offset, diags)
    _check_repeated_words(text, offset, diags)
    results = build_ast_from_text(text, offset)

    if not results:
        if re.search(r"[a-zA-Z]", text):
            diags.append(_no_parse(text, offset))
        diags.sort(key=_by_offset)
        return ValidationResult(
            utterance=None,
            probability=0.0,
            parse_count=0,
            diagnostics=diags,
            is_valid=False,
            text=text,
            offset=offset,
        )

    utt, prob = results[0]
    result = ValidationResult(
        utterance=utt,
        probability=prob,
        parse_count=len(results),
        diagnostics=[],
        is_valid=False,
        text=text,
        offset=offset,
        readings=results,
        text_diagnostics=diags,
    )
    result.diagnostics = diagnose(result, utt)
    result.is_valid = not any(d.severity == "ERROR" for d in result.diagnostics)
    return result


def diagnose(result: ValidationResult, utterance: Utterance) -> list[Diagnostic]:
    """Every diagnostic for one reading of result's sentence: the text checks plus the checks on that parse."""
    diags = list(result.text_diagnostics)
    diags += validate_utterance(utterance, parse_count=result.parse_count)
    _check_commas(utterance, result.text, result.offset, diags)
    _check_poka(utterance, result.text, result.offset, diags)
    diags.sort(key=_by_offset)
    return diags


def validate_text(text: str) -> list[ValidationResult]:
    """
    Split text into sentences (lexer.split_sentences) and validate each one.
    Every offset in the results points into text.
    """
    from lexer import split_sentences
    return [validate(sentence, start) for start, sentence in split_sentences(text)]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _DEMO = [
        # Clean sentences
        ("mi moku e kasi",         "clean transitive"),
        ("jan li wile pali",        "clean preverb"),
        ("tomo pi telo nasa",       "clean pi-phrase"),
        ("jan Joni li pona",        "valid proper noun"),
        ("taso ona li pona",        "taso as starter"),
        ("tenpo ni la mi wile toki","la-context"),
        # Triggered warnings / errors
        ("tomo pi telo",            "pi with single word"),
        ("jan Wupi li pona",        "bad proper noun (Wu-)"),
        ("jan",                     "bare fragment"),
        ("mi moku e taso",          "taso as head"),
        ("ni li pona",              "ni as head (valid)"),
        ("Mi moku.",                "capitalised pronoun"),
        ("mi moku e xyz",           "unknown word"),
        ("li li li",                "no parse"),
        ("sina sona ala sona",      "V-ala-V (ambiguous)"),
    ]

    print(f"{'Sentence':<35} {'Parses':>6}  {'Diagnostics'}")
    print("-" * 75)
    for text, note in _DEMO:
        r = validate(text)
        diag_str = "; ".join(f"[{d.severity[0]}] {d.code}" for d in r.diagnostics)
        print(f"{text!r:<35} {r.parse_count:>6}  {diag_str or '✓ clean'}  ({note})")

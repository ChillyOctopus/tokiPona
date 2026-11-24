"""
ast_nodes.py
Toki Pona AST node types.

All nodes are plain @dataclass (mutable). All carry a `source` string
containing content words only — structural particles (li, e, pi, la, en,
nanpa, o, anu) are stripped from source text.

Design: specific dataclasses, one per grammatical construct.  Python 3.10+
match/case structural pattern matching works directly against these types.

Pipeline position: lexer → parser → **AST builder** → validator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union


# ---------------------------------------------------------------------------
# Terminal leaf: a single word
# ---------------------------------------------------------------------------

@dataclass
class Word:
    """
    A single Toki Pona word as produced by the Lark lexer.

    token_type mirrors the Lark terminal name (WORD, PREV, PREP, NUM, UNMK,
    MOD, ALA, TASO, SEME, A) so callers can check lexer category without
    importing the lexer.
    """
    text:       str   # original text (capitalisation preserved)
    token_type: str   # Lark terminal name
    offset:     int   # character offset in original input (0-indexed)
    source:     str   # == text (redundant but keeps interface uniform)

    def __repr__(self) -> str:
        return f"Word({self.text!r}, {self.token_type})"


# ---------------------------------------------------------------------------
# Number
# ---------------------------------------------------------------------------

@dataclass
class Number:
    """
    A numeric phrase: one or more NUM/ALA tokens.
    Example: 'wan tu luka' → Number(words=[Word('wan'), Word('tu'), Word('luka')])
    """
    words:  list[Word]
    source: str


# ---------------------------------------------------------------------------
# Phrase-level nodes (noun / content phrases)
# ---------------------------------------------------------------------------

@dataclass
class SimplePhrase:
    """
    Head word plus zero or more modifier words.  No pi grouping, no ordinal.
    Example: 'jan pona' → SimplePhrase(head=Word('jan'), modifiers=[Word('pona')])
    """
    head:      Word
    modifiers: list[Word]
    source:    str


@dataclass
class OrdinalPhrase:
    """
    A simple phrase followed by 'nanpa <number>'.
    Example: 'jan nanpa wan' → OrdinalPhrase(base=SP('jan'), ordinal=Number([wan]))
    """
    base:    SimplePhrase
    ordinal: Number
    source:  str


@dataclass
class PiPhrase:
    """
    A phrase regrouped by one or more 'pi' markers.
    pi_groups is a flat list — one entry per 'pi' (not nested).

    'tomo pi telo nasa'        → PiPhrase(base=SP('tomo'), pi_groups=[SP('telo nasa')])
    'tomo pi telo pi nasa mute'→ PiPhrase(base=SP('tomo'), pi_groups=[SP('telo'), SP('nasa mute')])
    """
    base:      SimplePhrase | OrdinalPhrase
    pi_groups: list[SimplePhrase | OrdinalPhrase]   # 1+
    source:    str


# Union aliases — used throughout to type-hint "any phrase node"
AnyPhrase  = Union[SimplePhrase, OrdinalPhrase, PiPhrase]
NounPhrase = Union[SimplePhrase, OrdinalPhrase, PiPhrase, "DisjunctPhrase"]


@dataclass
class DisjunctPhrase:
    """
    Two or more phrases joined by 'anu' at the phrase level.
    Only created when there are 2+ options; single phrase passes through unchanged.
    """
    options: list[AnyPhrase]   # 2+
    source:  str


# ---------------------------------------------------------------------------
# Preposition and its phrase
# ---------------------------------------------------------------------------

@dataclass
class Preposition:
    """
    A preposition token, optionally negated ('lon ala') and/or emphasised ('lon a').
    Grammar allows PREP [A] [ALA] [A], or PREP ala PREP to ask a yes/no question.
    """
    word:     Word
    negated:  bool
    emphasis: bool
    source:   str
    question: bool = False   # 'lon ala lon' (V ala V)


@dataclass
class PrepPhrase:
    """
    A preposition plus its noun-phrase argument.
    Example: 'lon tomo' → PrepPhrase(prep=Preposition(Word('lon')), obj=SP('tomo'))
    """
    prep:   Preposition
    obj:    NounPhrase
    source: str


# ---------------------------------------------------------------------------
# Object phrase (e-marked)
# ---------------------------------------------------------------------------

@dataclass
class ObjectPhrase:
    """
    A direct object: 'e <NP>' plus any prep phrases immediately following it.
    Example: 'e tomo lon ma' → ObjectPhrase(np=SP('tomo'), preps=[PP('lon','ma')])
    """
    np:     NounPhrase
    preps:  list[PrepPhrase]
    source: str


# ---------------------------------------------------------------------------
# Preverb and its phrase
# ---------------------------------------------------------------------------

@dataclass
class Preverb:
    """
    A single preverb word, optionally negated ('wile ala') and/or emphasised.
    Grammar allows PREV [A] [ALA] [A], or PREV ala PREV to ask a yes/no question.
    """
    word:     Word
    negated:  bool
    emphasis: bool
    source:   str
    question: bool = False   # 'wile ala wile' (V ala V)


# ---------------------------------------------------------------------------
# Verb phrase variants
# ---------------------------------------------------------------------------

@dataclass
class VerbPhraseTransitive:
    """
    Verb phrase with at least one direct object.
    'moku e kasi' → VPT(verb=SP('moku'), objects=[OP(SP('kasi'))], preps=[])
    """
    verb:    AnyPhrase
    objects: list[ObjectPhrase]   # 1+
    preps:   list[PrepPhrase]     # optional trailing prepositions
    source:  str


@dataclass
class VerbPhraseIntransitive:
    """
    Verb phrase with no direct objects (may have prep phrases).
    'toki tawa mi' → VPI(verb=SP('toki'), preps=[PP('tawa', SP('mi'))])
    """
    verb:  AnyPhrase
    preps: list[PrepPhrase]
    source: str


@dataclass
class VerbPhrasePrepositional:
    """
    A predicate that is purely prepositional — no overt verb word.
    'lon tomo kepeken noka' → VPPrep(preps=[PP('lon','tomo'), PP('kepeken','noka')])
    """
    preps:  list[PrepPhrase]   # 1+
    source: str


@dataclass
class VerbPhraseYN:
    """
    V-ala-V yes/no question form.
    'sona ala sona'        → VPYN(verb=SP('sona'), repeated_verb=SP('sona'), objects=[])
    'sona ala sona e toki' → VPYN(verb=SP('sona'), repeated_verb=SP('sona'), objects=[OP(SP('toki'))])
    'pona ala pona tawa sina' → VPYN(..., preps=[PP('tawa', SP('sina'))])
    repeated_verb is the phrase after 'ala'; stored for mismatch validation.
    """
    verb:          AnyPhrase
    repeated_verb: AnyPhrase | None   # the phrase after ALA (should match verb.head.text)
    objects:       list[ObjectPhrase]
    source:  str
    preps:         list[PrepPhrase] = field(default_factory=list)


# Union alias for all verb phrase types
VerbPhrase = Union[
    VerbPhraseTransitive,
    VerbPhraseIntransitive,
    VerbPhrasePrepositional,
    VerbPhraseYN,
]


@dataclass
class PreverbPhrase:
    """
    One or more preverbs followed by a verb phrase.
    Flattened from the right-recursive grammar rule.

    'wile ala pali'    → PreverbPhrase([Preverb('wile',negated=True)], VPI('pali'))
    'ken wile toki'    → PreverbPhrase([PV('ken'), PV('wile')], VPI('toki'))
    """
    preverbs: list[Preverb]   # 1+
    vp:       VerbPhrase
    source:   str


# A predicate slot is either a full preverb phrase or a bare verb phrase
Predicate = Union[PreverbPhrase, VerbPhrase]


# ---------------------------------------------------------------------------
# Subject types
# ---------------------------------------------------------------------------

@dataclass
class UnmarkedSubject:
    """mi or sina — subjects that appear without 'li'."""
    word:   Word
    source: str


@dataclass
class MarkedSubject:
    """Any noun phrase that precedes 'li'."""
    phrase: NounPhrase
    source: str


@dataclass
class CompoundSubject:
    """
    Two or more subjects joined by 'en' or 'anu'.
    'mi en sina' → CompoundSubject([SP('mi'), SP('sina')], 'en')
    """
    parts:       list[NounPhrase]   # 2+
    conjunction: Literal["en", "anu"]
    source:      str


Subject = Union[UnmarkedSubject, MarkedSubject, CompoundSubject]


# ---------------------------------------------------------------------------
# Clause and sentence-level nodes
# ---------------------------------------------------------------------------

@dataclass
class ContextPhrase:
    """
    A 'la' boundary: either a noun phrase ('tenpo ni la') or a full clause ('mi pona la').
    The LA particle itself is stripped.
    """
    content: Union[NounPhrase, Subject, "Clause"]
    source:  str


@dataclass
class Clause:
    """
    A single clause: optional subject + predicate list.

    predicate_type:
        'li'   — marked subject (phrase li predicates)
        'none' — unmarked subject mi/sina (no li)
        'o'    — deontic/optative (subject o predicates, or bare o predicates)

    subject is None only for bare-o clauses without an addressee.
    predicates is a flat list — serial predicates (li…li) each become one entry.
    """
    subject:        Subject | None
    predicate_type: Literal["li", "none", "o"]
    predicates:     list[Predicate]   # 1+ (empty only for bare subject fragments)
    source:         str


@dataclass
class Vocative:
    """
    A vocative address: 'NP o' with no following predicate.
    'jan pona o' → Vocative(addressee=MarkedSubject(SP('jan pona')))
    """
    addressee: NounPhrase | Subject
    source:    str


@dataclass
class SentenceStarter:
    """
    A sentence-initial discourse particle: 'taso', 'kin', 'a', 'taso a', etc.
    """
    words:  list[Word]   # 1–2 tokens
    source: str


@dataclass
class Sentence:
    """
    A full sentence with optional starters, zero or more la-contexts, one clause,
    and an optional 'anu seme' weak-question marker.

    vocative:    a call for attention before or after the sentence
                 ('jan Nowa o, soweli sina li wile musi!')
    elaboration: the sentence after a colon, often what 'e ni:' announces
                 ('ijo li toki e ni: suno li lon.')
    """
    starters:    list[SentenceStarter]
    contexts:    list[ContextPhrase]
    clause:      Clause
    anu_seme:    bool
    source:      str
    vocative:    Vocative | None = None
    elaboration: Sentence | None = None


@dataclass
class Utterance:
    """
    Root AST node.  The 'body' is either a Sentence or a bare Vocative.
    starters here are only used when the body is a Vocative (otherwise they
    live inside Sentence.starters).
    body is None for bare interjections ("a!", "o!") that have no sentence body.
    """
    starters: list[SentenceStarter]
    body:     Sentence | Vocative | None
    source:   str

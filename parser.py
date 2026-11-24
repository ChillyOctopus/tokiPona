"""
parser.py
Toki Pona Earley parser.

API:
    parse(text: str) -> list[ParseResult]
        Returns all grammatical interpretations of the input, ranked by score.
        Returns [] for empty input or completely ungrammatical input.
        Input is ONE sentence. Dictionary words are lowercased first
        (lexer.normalize), so in "Mi moku." mi is the pronoun, not a name.

    explain_failure(text: str) -> tuple[int | None, str] | None
        Why a sentence does not parse: (character offset or None, message).

    ParseResult.tree  — Lark Tree; node labels are grammar rule names
    ParseResult.score — raw additive score (higher = preferred)
    ParseResult.probability — softmax-normalised probability across all parses

Grammar synthesised from:
    ilo_nasin_sin.cfg     — primary clause structure, preverb/prep negation
    ztomaszewski.cfg      — V ala V (Y/N) question form
    nasin_alesa.cfg       — anu seme weak question; preferred/deprioritise strategy
    toki_pona_parser_master.cfg — sentence starters, multiple question types
    wasona.cfg            — calls before/after a sentence, colons, V ala V on
                            preverbs and prepositions, ranking preferences
    jan_lope_robert.cfg   — lexical sets (source of record for word lists)

Run tests: .venv/bin/python3 -m pytest tests/test_parser.py -v
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lark import Lark, Token, UnexpectedEOF, UnexpectedInput, UnexpectedToken
from lark.tree import Tree

from lexer import normalize

# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------
#
# TERMINAL PRIORITY DESIGN
# ========================
# WORD.1   — lowest: catch-all for all content words (nouns, verbs, modifiers)
# LI/E/PI/etc. .2 — structural particles: must beat WORD.1
# PREP/PREV/NUM/UNMK/MOD .3 — word categories: beat structural particles
# ALA.4    — highest among single words: prevents Lark's dynamic lexer from
#             ever splitting "ala" into the "a" particle + "la" particle.
#             ALA is kept separate from NUM so its interactions are explicit.
#
# LEXER='basic'
# =============
# Deterministic single-pass tokenisation using longest match + priority.
# This prevents the Lark dynamic/contextual lexer from exploring alternative
# tokenisations of individual words (e.g. "ala" → "a"+"la").

_GRAMMAR = r"""
// ============================================================
// TOKI PONA GRAMMAR  (Earley / basic-lexer / ambiguity=explicit)
// ============================================================

// --- TOP LEVEL ---

// A call for attention can come before the sentence ("jan Nowa o, soweli
// sina li wile musi!"), after it ("o toki ala e wawa, jan nasa o.") or stand
// alone ("jan Ema o!") — wasona lesson 12 and more/28-02.
start: sentence_starter* vocative? sentence vocative?
     | sentence_starter* vocative
     | sentence_starter+
     | O A*

// Only canonical sentence starters — prevents content words like "tenpo" from
// being consumed as starters. N is the hesitation "n" (wasona lesson 28).
// ANU/EN added for discourse-initial continuations ("anu jaki!", "en moku").
sentence_starter: TASO A?
               | MOD  A?
               | A
               | N
               | ANU
               | EN

// --- SENTENCE ---

// "anu seme" tags a statement and a colon follows one (wasona lessons 17 and
// 10), so both need a full clause, not a bare phrase ("sina anu seme?" is wrong)
sentence: context_clause* clause_any
        | context_clause* clause_strict ANU SEME
        | context_clause* clause_strict COLON elaboration?

// What a colon announces: "ijo li toki e ni: suno li lon." (wasona lesson 10).
// A colon may also end the sentence; the content then comes as the next one.
elaboration: sentence_starter* sentence

context_clause: general_subject LA
             | clause_strict    LA

// --- VOCATIVE ---

vocative: general_subject O
        | general_subject O A+

// --- CLAUSE VARIANTS ---

// clause_any: bare subject fragment allowed (no predicate required)
clause_any: general_subject
          | marked_subject   LI predicate_list_li
          | unmarked_subject    predicate_list_none
          | general_subject  O  predicate_list_o
          |                  O  predicate_list_o

// clause_strict: full predicate required (used inside context la)
clause_strict: marked_subject   LI predicate_list_li
             | unmarked_subject    predicate_list_none
             | general_subject  O  predicate_list_o
             |                  O  predicate_list_o

// --- PREDICATE LISTS (serial predicates) ---

predicate_list_li: preverb_phrase
                 | preverb_phrase (LI preverb_phrase | O preverb_phrase | ANU preverb_phrase)+

predicate_list_none: preverb_phrase
                   | preverb_phrase ((LI | O | ANU) preverb_phrase)+

predicate_list_o: preverb_phrase
                | preverb_phrase (O preverb_phrase)+

// --- SUBJECTS ---

general_subject: marked_subject
              | unmarked_subject

marked_subject: phrase
             | compound_subject

compound_subject: phrase EN phrase
               | phrase EN compound_subject
               | phrase ANU phrase
               | phrase ANU compound_subject

// "sina a moku e moku!" — a emphasises the word before it (wasona lesson 18)
unmarked_subject: UNMK
                | UNMK A

// --- PREVERB PHRASE ---

preverb_phrase: verb_phrase
             | preverb preverb_phrase
             | preverb_yn preverb_phrase

// V ala V on a preverb asks a yes/no question: "sina wile ala wile lape?"
// (wasona lesson 17 doubles the first word of the verb, even a preverb)
preverb_yn: PREV ALA PREV

preverb: PREV
       | PREV A
       | PREV ALA
       | PREV A ALA
       | PREV ALA A
       | PREV A ALA A

// --- VERB PHRASE ---

verb_phrase: verb_phrase_transitive
           | verb_phrase_intransitive
           | verb_phrase_prepositional
           | verb_phrase_yn

// Transitive: phrase + at least one e-object
verb_phrase_transitive: phrase object_phrase+ prep_phrase*

// Intransitive: phrase + optional prepositional phrases
verb_phrase_intransitive: phrase
                        | phrase prep_phrase+

// Prepositional-only (no overt verb)
verb_phrase_prepositional: prep_phrase+

// V ala V: yes/no question (ztomaszewski / toki_pona_parser_master; wasona lesson 17)
verb_phrase_yn: phrase ALA phrase object_phrase* prep_phrase*

// --- OBJECTS AND PREPOSITIONS ---

object_phrase: E disjunct_phrase
             | E disjunct_phrase prep_phrase+

prep_phrase: preposition disjunct_phrase
           | preposition_yn disjunct_phrase

// V ala V on a preposition: "ona li lon ala lon ma Mewika?" (wasona lesson 17)
preposition_yn: PREP ALA PREP

preposition: PREP
           | PREP A
           | PREP ALA
           | PREP A ALA
           | PREP ALA A
           | PREP A ALA A

// --- NOUN / CONTENT PHRASES ---

// Disjunction: phrase anu phrase anu ...
disjunct_phrase: phrase
              | phrase ANU phrase
              | phrase (ANU phrase)+

phrase: pi_phrase

// pi-phrase: one group per pi, left-associated ("tomo pi telo nasa pi jan mute").
// A pi group is a word and its modifiers, never another pi group (wasona
// lesson 22), so "pi telo pi nasa mute" can't hide a pi before one word.
pi_phrase: nanpa_phrase
         | pi_phrase PI nanpa_phrase

// Ordinal: simple_phrase [nanpa number]
nanpa_phrase: simple_phrase
            | simple_phrase NANPA number

// Core noun phrase: head + zero or more modifiers
simple_phrase: head
             | head modifier+

// Heads: any content-bearing token
head: WORD
    | PREV
    | PREP
    | NUM
    | NUM_ABBR
    | UNMK
    | NANPA
    | MOD
    | ALA
    | TASO
    | SEME

// Modifiers: same as heads plus emphasis "a"
// UNMK (mi/sina) added for possessive/pronoun modifiers: "tomo mi", "olin sina".
// NANPA added for use as a content noun: "nanpa ni li ale", "ilo nanpa".
modifier: WORD
        | PREV
        | PREP
        | NUM
        | NUM_ABBR
        | UNMK
        | NANPA
        | MOD
        | ALA
        | A
        | TASO
        | SEME

// Number: one or more numeric words; ala (0) may appear in the sequence
number: NUM+
      | NUM_ABBR
      | ALA
      | ALA NUM+
      | NUM+ ALA
      | NUM+ ALA NUM+

// ============================================================
// TERMINAL SETS
// ============================================================
//
// Priority levels:
//   ALA.4  — "ala" specifically; prevents splitting into "a"+"la"
//   PREP.3 / PREV.3 / NUM.3 / UNMK.3 / MOD.3  — word categories
//   LI.2 / E.2 / PI.2 / LA.2 / EN.2 / O.2 / ANU.2 / A.2 / NANPA.2
//             — structural particles
//   TASO.2 / SEME.2  — sentence-starter / weak-question marker
//   WORD.1 — catch-all for everything else
//
// WORD BOUNDARY RULE:
// All terminals use (?![a-z]) so they only match at a word boundary.
// This prevents "o" from consuming the "o" in "ona", "li" from "lili", etc.
// Capitalised-word MOD.3 never has this issue (uppercase start is unambiguous).
// ============================================================

// "ala" — negation / V-ala-V / numeral zero — always one token
ALA.4:   /ala(?![a-z])/

// Word categories — word-boundary anchored
PREP.3:  /(?:kepeken|lon|sama|tan|tawa)(?![a-z])/
PREV.3:  /(?:awen|kama|ken|lukin|open|pini|sona|wile|alasa)(?![a-z])/
NUM.3:   /(?:ale|ali|luka|mute|tu|wan)(?![a-z])/
UNMK.3:  /(?:mi|sina)(?![a-z])/
MOD.3:   /kin(?![a-z])|[A-Z][a-zA-Z]*/

// Structural particles — word-boundary anchored to prevent prefix stealing
// (e.g. O would consume "o" from "ona" without the lookahead)
LI.2:    /li(?![a-z])/
E.2:     /e(?![a-z])/
PI.2:    /pi(?![a-z])/
LA.2:    /la(?![a-z])/
EN.2:    /en(?![a-z])/
O.2:     /o(?![a-z])/
ANU.2:   /anu(?![a-z])/
A.2:     /a(?![a-z])/
NANPA.2: /nanpa(?![a-z])/

// Sentence-starter words and weak-question marker
TASO.2:  /taso(?![a-z])/
SEME.2:  /seme(?![a-z])/

// The hesitation "n" (a sentence starter only) and the colon, the one
// punctuation mark the grammar reads: it ties a sentence to what it announces.
N.2:     /n(?![a-z])/
COLON.2: /:/

// wasona's Latin shorthand for numbers: W T L M A = wan tu luka mute ale
// after a # ("tenpo mun #LTT" = luka tu tu; more/25-01)
NUM_ABBR.4: /#[WTLMA]+(?![a-zA-Z])/

// Catch-all for nouns, verbs, modifiers, mu, etc.
WORD.1:  /[a-z]+/

%ignore /\s+/
%ignore /[^a-zA-Z\s:]+/
"""

# Build parser once at module load.
# lexer='basic'  — single-pass deterministic tokenisation; prevents "ala" → "a"+"la"
# ambiguity='explicit' — _ambig nodes wrap alternatives; enumerated in parse()
_PARSER = Lark(
    _GRAMMAR,
    parser="earley",
    lexer="basic",
    ambiguity="explicit",
    maybe_placeholders=False,
)

# ---------------------------------------------------------------------------
# Scoring  (ported from ilo-nasin-sin parser.ts)
# ---------------------------------------------------------------------------

# Rule names that indicate well-structured / preferred parses.
_GOOD_RULES: frozenset[str] = frozenset({
    "preverb_phrase",
    "verb_phrase_transitive",
    "verb_phrase_intransitive",
    "verb_phrase_prepositional",
    "verb_phrase_yn",
    "prep_phrase",
    "object_phrase",
    "compound_subject",
    "predicate_list_li",
    "predicate_list_none",
    "predicate_list_o",
    "context_clause",
    "vocative",
    "preverb",
    "preverb_yn",
    "preposition_yn",
})

# Words rarely/never acting as phrase heads — penalised when in head position.
_BAD_HEADS: frozenset[str] = frozenset({"taso", "kin", "n"})

_GOOD_WEIGHT: float =  1.0
_BAD_WEIGHT:  float = -1.5

# Rule-specific preferences between competing readings (see _preference).
_YN_MATCH:         float =  3.0   # V ala V repeats the same word...
_YN_MISMATCH:      float = -4.0   # ...so a different word means it isn't V ala V
_ANU_SEME_TAG:     float =  3.0   # "anu seme" is a question tag, not a second verb
_MI_X_IS_SUBJECT:  float = -3.0   # "mi mute li/o ...": the subject is mi mute (wasona lesson 16)
_COMMA_AFTER_CALL: float =  2.0   # "jan Nowa o, ...": a call, then a sentence
_COMMA_IN_PHRASE:  float = -2.0   # commas fall between phrases, not inside one
_FRAGMENT:         float = -1.0   # a clause with no verb is a last resort
_PI_ONE_WORD:      float = -3.0   # pi never comes before a single word (wasona lesson 23)

_PHRASE_RULES: frozenset[str] = frozenset({"simple_phrase", "pi_phrase"})


def score_tree(tree: Tree, text: str = "") -> float:
    """
    Recursively score a Lark parse tree.
    Returns a raw additive score (higher = preferred parse).
    Penalises parses where structural/discourse words appear as phrase heads.
    text is the sentence that was parsed: commas are not part of the grammar,
    but they mark likely boundaries between phrases.
    """
    return _score(tree, tuple(i for i, ch in enumerate(text) if ch == ","))


def _score(tree: Tree, commas: tuple[int, ...]) -> float:
    score = _GOOD_WEIGHT if tree.data in _GOOD_RULES else 0.0

    # Penalise bad words used as phrase heads.
    if tree.data == "head":
        for child in tree.children:
            # kin and taso are never heads (wasona): "kala li kama kin" is
            # verb kama + kin, not preverb kama + verb kin
            if hasattr(child, "value") and child.value in _BAD_HEADS:
                score += _BAD_WEIGHT * 2
            # ala is rarely a head: "mi sona ala" is "I don't know",
            # not preverb sona + verb ala
            if hasattr(child, "type") and child.type == "ALA":
                score += _BAD_WEIGHT * 2
            # A name is never a head (wasona more/07-01): "tomo tawa Ninja" is
            # a car named Ninja, not "a house, to Ninja"
            if getattr(child, "type", None) == "MOD" and str(child)[:1].isupper():
                score += _BAD_WEIGHT * 2

    # Penalise NANPA appearing as a modifier (prefer the ordinal-phrase reading).
    # When "X nanpa N" can parse as either SimplePhrase(X, [nanpa, N]) or
    # OrdinalPhrase via nanpa_phrase, the ordinal parse is linguistically correct.
    if tree.data == "modifier":
        for child in tree.children:
            if hasattr(child, "type") and child.type == "NANPA":
                score += _BAD_WEIGHT * 0.75

    score += _preference(tree, commas)

    # Recurse into child subtrees.
    for child in tree.children:
        if isinstance(child, Tree):
            score += _score(child, commas)

    return score


def _tokens(tree: Tree) -> list[Token]:
    return list(tree.scan_values(lambda v: isinstance(v, Token)))


def _first_word(node) -> str:
    """The first word of a subtree (a phrase's head), or the token itself."""
    return str(node if isinstance(node, Token) else _tokens(node)[0])


def _preference(tree: Tree, commas: tuple[int, ...]) -> float:
    """Rule-specific preferences between competing readings, mostly from wasona."""
    kind, children = tree.data, tree.children
    if kind == "nanpa_phrase" and len(children) > 1:
        return _GOOD_WEIGHT   # an ordinal ("jan nanpa wan"); every phrase passes through nanpa_phrase
    if kind in ("verb_phrase_yn", "preverb_yn", "preposition_yn"):
        same = _first_word(children[0]) == _first_word(children[2])
        return _YN_MATCH if same else _YN_MISMATCH
    if kind == "sentence" and any(getattr(c, "type", None) == "SEME" for c in children):
        return _ANU_SEME_TAG
    if kind == "clause_any" and len(children) == 1:
        return _FRAGMENT
    if kind == "pi_phrase" and len(children) == 3 and len(_tokens(children[2])) == 1:
        return _PI_ONE_WORD
    if (kind == "predicate_list_none" and len(children) > 1
            and getattr(children[1], "type", None) in ("LI", "O") and len(_tokens(children[0])) == 1):
        return _MI_X_IS_SUBJECT
    if kind == "vocative":
        o = next(c for c in children if getattr(c, "type", None) == "O")
        return _COMMA_AFTER_CALL if any(o.end_pos <= c <= o.end_pos + 1 for c in commas) else 0.0
    if kind in _PHRASE_RULES and commas:
        tokens = _tokens(tree)
        if any(tokens[0].end_pos <= c < tokens[-1].start_pos for c in commas):
            return _COMMA_IN_PHRASE
    return 0.0


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Tree enumeration (expand _ambig nodes → concrete parse trees)
# ---------------------------------------------------------------------------

def _enumerate_trees(tree: Tree):
    """
    Yield every concrete parse tree from a Lark result that may contain
    _ambig nodes.  _ambig nodes represent local choice points; we take the
    Cartesian product of all child alternatives.
    """
    from itertools import product as iproduct

    if tree.data == "_ambig":
        for child in tree.children:
            if isinstance(child, Tree):
                yield from _enumerate_trees(child)
            else:
                yield child
        return

    child_variants: list[list] = []
    for child in tree.children:
        if isinstance(child, Tree):
            child_variants.append(list(_enumerate_trees(child)))
        else:
            child_variants.append([child])

    for combo in iproduct(*child_variants):
        yield Tree(tree.data, list(combo))


# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParseResult:
    score:       float  # raw additive score
    probability: float  # softmax-normalised probability (0–1) across all parses
    tree:        Tree   # Lark parse tree (data = rule name, children = subtrees/tokens)

    def __repr__(self) -> str:
        return (
            f"ParseResult(score={self.score:.2f}, "
            f"probability={self.probability:.3f}, "
            f"tree_root={self.tree.data!r})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(text: str) -> list[ParseResult]:
    """
    Parse a Toki Pona string and return all grammatical interpretations,
    sorted by score (highest first).

    Returns [] for empty input or completely ungrammatical input.
    Each ParseResult carries a .tree (Lark Tree), .score, and .probability.

    The grammar is intentionally ambiguous — most sentences produce 1–5 parses.
    Scoring selects the most linguistically natural interpretation at index 0.
    """
    if not text.strip():
        return []

    try:
        result = _PARSER.parse(normalize(text))
    except UnexpectedInput:
        return []

    # Expand all _ambig nodes into concrete trees and deduplicate.
    seen: set[str] = set()
    trees: list[Tree] = []
    for t in _enumerate_trees(result):
        key = t.pretty()
        if key not in seen:
            seen.add(key)
            trees.append(t)

    raw_scores = [score_tree(t, text) for t in trees]
    probs = _softmax(raw_scores)

    return sorted(
        [ParseResult(score=s, probability=p, tree=t)
         for s, p, t in zip(raw_scores, probs, trees)],
        key=lambda r: r.score,
        reverse=True,
    )


def explain_failure(text: str) -> tuple[int | None, str] | None:
    """
    Say why a sentence does not parse: (character offset or None, message).
    The offset points at the first word the grammar cannot accept; it is None
    when the sentence stops too early. Returns None if the text parses.
    """
    try:
        _PARSER.parse(normalize(text))
    except UnexpectedToken as err:
        if err.token.type == "EN":
            return err.token.start_pos, ("'en' only joins subjects, not words in an "
                                         "object or a pi group (wasona lesson 26)")
        return err.token.start_pos, f"'{err.token}' is not expected here"
    except UnexpectedEOF:
        return None, "the sentence ends before it is complete"
    except UnexpectedInput as err:
        return err.pos_in_stream, "the sentence cannot be parsed"
    return None


def pretty(tree: Tree, indent: int = 0) -> str:
    """Format a Lark parse tree as an indented string for debugging."""
    pad = "  " * indent
    if not isinstance(tree, Tree):
        return f"{pad}{tree!r}"
    children_str = "\n".join(pretty(c, indent + 1) for c in tree.children)
    if children_str:
        return f"{pad}({tree.data}\n{children_str})"
    return f"{pad}({tree.data})"


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _DEMO = [
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
    ]

    for sentence in _DEMO:
        results = parse(sentence)
        top = results[0] if results else None
        print(f"\n{'=' * 60}")
        print(f"  {sentence!r}  →  {len(results)} parse(s)")
        if top:
            print(f"  Top: score={top.score:.1f}  p={top.probability:.3f}")
            print(pretty(top.tree, indent=2))

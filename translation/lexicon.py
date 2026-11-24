"""
translation/lexicon.py
The English a Toki Pona word can become, by the role it plays in a sentence:
noun, verb, adjective (Toki Pona modifiers, which also serve as adverbs) or
preposition.

Sources, best first:
  1. wasona's glosses in wasona_sentences.tsv: the one-word drills (kind
     "vocab") and the word lists (kind "word"). wasona is the authority, and its
     English is what the evaluation compares against.
  2. master_dictionary.yaml: each sense's part of speech and gloss, ranked by
     ku_data, the share of speakers who translate the word that way.

A gloss gives English without saying its role ("pona: good, pleasant"). An
item takes the role the word's dictionary senses give it; if they don't list
it, the role any other word's senses give it; if none do, noun. An item with
several roles ("good" is a noun and an adjective) takes the one the word has
the most senses for.

Two tables beside this file add what single words can't: compounds.tsv
(multi-word expressions such as "telo nasa" = alcohol, and fixed contexts
before la) and names.tsv (the English for names: "ma Tosi" = Germany).
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from lexer import ENTRIES

_HERE = Path(__file__).parent
_WASONA_TSV = _HERE.parent / "wasona_sentences.tsv"

ROLES = ("noun", "verb", "adjective", "preposition")

# master_dictionary.yaml sense tags -> roles
_POS_ROLES: dict[str, str] = {
    "n": "noun", "m": "adjective", "v": "verb", "vt": "verb", "vi": "verb", "p": "preposition",
}

# When a word has as many senses in two roles, the earlier role wins
_TIE_ORDER = ("adjective", "verb", "noun", "preposition")

# Endings that mark an English adjective no dictionary sense lists ("amazing")
_ADJECTIVE_ENDING = re.compile(r"(?:ing|ful|ous|ive|able|ible|ic|ish|less|ant|ent|y)$")

# wasona lesson 5 tags some glosses: "N: food; V: to consume, to eat"
_TAGS: dict[str, str] = {"N": "noun", "V": "verb", "Adj": "adjective"}

# Ranks: wasona's one-word drill, then its word lists, then the dictionary
# (ku_data share plus a little for an early position).
_DRILL, _WORD_LIST = 4.0, 3.0

# Corrections the evaluation turned up; they win over the computed values.
_OVERRIDES: dict[str, dict] = {
    "soweli": {"noun": ["animal"]},               # the drill says "land animal"; the sentences say "animal"
    "kalama": {"verb": ["make a sound"]},         # wasona: "to produce a sound"
    "weka": {"predicate": "adjective", "adjective": ["away", "gone"]},   # "sona li weka": knowledge is gone
}


@dataclass
class Senses:
    """The English one Toki Pona word can become."""
    word: str
    roles: dict[str, list[str]] = field(default_factory=dict)   # role -> options, best first
    predicate: str = "noun"   # its role as a bare predicate: "waso li pona" -> adjective

    def options(self, role: str) -> list[str]:
        return self.roles.get(role, [])


@dataclass
class Lexicon:
    words: dict[str, Senses]
    compounds: dict[str, dict[tuple[str, ...], str]]   # role -> words -> English
    names: dict[str, tuple[str, str]]                  # name -> (English, language): "Tosi" -> ("Germany", "German")
    utterances: dict[str, str]                         # "toki!" -> "Hello! Let's talk!"

    def get(self, word: str) -> Senses | None:
        return self.words.get(word)


def _clean(gloss: str) -> str:
    """Drop examples, glyphs, percentages and parentheses from a gloss."""
    gloss = re.sub(r",?\s*\blike (?:a |an )?[^;]*", "", gloss)   # "tool, like a hammer, fork"
    gloss = re.sub(r"\([^)]*\)", "", gloss)
    return re.sub(r"[^\x00-\x7f]", "", gloss)


def _items(gloss: str) -> list[str]:
    """'to produce a sound; sound' -> ['to produce a sound', 'sound']."""
    out = []
    for part in re.split(r"[;,/]", _clean(gloss)):
        if "..." in part:   # a preverb frame: "to want to ..."
            continue
        part = part.strip(" .!?'\"").lower()
        if part and not part.startswith("-") and not part.endswith("-") and len(part.split()) <= 3:
            out.append(part)
    return out


def _normalise(item: str, role: str) -> str:
    if role == "verb":
        item = re.sub(r"^to ", "", item)
    if role == "noun":
        item = re.sub(r"^(?:a|an|the) ", "", item)
    return item


def _sense_items(entry: dict):
    """(role, English, pos) for every item of every sense of a dictionary entry."""
    for sense in entry.get("senses", []):
        role = _POS_ROLES.get(sense.get("pos", ""))
        if role:
            for item in _items(sense.get("gloss", "")):
                yield role, _normalise(item, role), sense["pos"]


@lru_cache(maxsize=1)
def _english_roles() -> dict[str, frozenset[str]]:
    """Every English item in the dictionary, with the roles its senses give it."""
    roles: dict[str, set[str]] = {}
    for entry in ENTRIES.values():
        for role, item, _ in _sense_items(entry):
            roles.setdefault(item, set()).add(role)
    return {item: frozenset(r) for item, r in roles.items()}


def _wasona_glosses() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    """wasona's drills and word-list glosses by word, and its one-word utterances."""
    drills: dict[str, list[str]] = {}
    lists: dict[str, list[str]] = {}
    utterances: dict[str, str] = {}
    with open(_WASONA_TSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            tp, en = row["toki_pona"].strip(), row["english"].strip()
            if not en:
                continue
            if row["kind"] == "vocab":
                drills.setdefault(tp, []).append(en)
            elif row["kind"] == "word" and re.fullmatch(r"[a-z]+[.!]", tp):
                utterances[tp] = en   # "toki!" -> "Hello! Let's talk!"
            elif row["kind"] == "word" and re.fullmatch(r"[a-z]+", tp):
                lists.setdefault(tp, []).append(en)
    return drills, lists, utterances


class _Builder:
    """Collects one word's options from the dictionary and wasona, then ranks them."""

    def __init__(self, entry: dict):
        self.known: dict[str, list[str]] = {r: [] for r in ROLES}   # the word's own items by role
        self.transitive: set[str] = set()     # verbs only listed with e (vt)
        self.intransitive: set[str] = set()   # verbs listed without it (v, vi)
        self.weights = Counter()              # senses per role
        self.scores: dict[tuple[str, str], float] = {}
        self.ku = {re.sub(r"\s*\(.*\)$", "", k).lower(): v for k, v in entry.get("ku_data", {}).items()}
        for sense in entry.get("senses", []):
            if sense.get("pos") in _POS_ROLES:
                self.weights[_POS_ROLES[sense["pos"]]] += 1
        for role, item, pos in _sense_items(entry):
            if item not in self.known[role]:
                self.known[role].append(item)
            if role == "verb":
                (self.transitive if pos == "vt" else self.intransitive).add(item)
        for role, items in self.known.items():
            for i, item in enumerate(items):
                self.add(role, item, self.ku.get(item, 0) / 100 + 0.5 / (i + 1))

    def add(self, role: str, item: str, score: float) -> None:
        if item:
            self.scores[(role, item)] = max(self.scores.get((role, item), 0.0), score)

    def known_role(self, item: str) -> str | None:
        """The role the dictionary gives an item: the word's own senses first,
        then any word's; with several, the one the word has most senses in."""
        roles = [r for r in ROLES if item in self.known[r]] or list(_english_roles().get(item, ()))
        if not roles:
            return None
        return max(roles, key=lambda r: (self.weights[r], -_TIE_ORDER.index(r)))

    def items(self, gloss: str) -> list[tuple[str, str]]:
        """(role, English) pairs from one wasona gloss, in its order. wasona puts
        one role between semicolons ("to have fun; funny, amusing"), so an item
        no sense lists takes its neighbours' role; failing that, an adjective
        ending ("-ing", "-ful") makes it an adjective, and anything else a noun."""
        tagged = re.findall(r"\b(N|V|Adj): ([^;]*)", gloss)
        if tagged:
            return [(_TAGS[t], _normalise(i, _TAGS[t])) for t, items in tagged for i in _items(items)]
        pairs = []
        for segment in _clean(gloss).split(";"):
            items = _items(segment)
            roles = ["verb" if i.startswith("to ") else self.known_role(i) for i in items]
            shared = next((r for r in roles if r), None)
            for item, role in zip(items, roles):
                if role == "verb" and item.startswith("to "):
                    item = item[3:]
                if role is None:
                    one_word_adjective = " " not in item and _ADJECTIVE_ENDING.search(item)
                    role = shared or ("adjective" if one_word_adjective else "noun")
                pairs.append((role, item))
        return pairs

    def learn_tags(self, glosses: list[str]) -> None:
        """wasona's role tags ("Adj: knowledgeable") teach the word's own roles."""
        for gloss in glosses:
            if re.search(r"\b(N|V|Adj): ", gloss):
                for role, item in self.items(gloss):
                    if item not in self.known[role]:
                        self.known[role].append(item)


def _senses(word: str, entry: dict, drills: list[str], lists: list[str]) -> Senses:
    b = _Builder(entry)
    b.learn_tags(lists)
    for rank, glosses in ((_WORD_LIST, lists), (_DRILL, drills)):
        for gloss in glosses:
            for i, (role, item) in enumerate(b.items(gloss)):
                b.add(role, item, rank - 0.01 * i)

    roles: dict[str, list[str]] = {}
    for (role, item), _ in sorted(b.scores.items(), key=lambda kv: -kv[1]):
        roles.setdefault(role, []).append(item)

    drill = b.items(drills[0])[:1] if drills else []
    # wasona glosses a verb ("kute: ear; to hear"); its "to do this" for ni,
    # ijo and seme (more/17-01) is a special use, not the word's own verb
    says_verb = any(i.startswith("to ") and not i.startswith("to do ")
                    for g in drills + lists for i in _items(g))
    top_verb = roles.get("verb", [None])[0]
    if drill and drill[0][0] == "adjective":
        predicate = "adjective"   # "pona: good"
    elif says_verb:
        # "weka: to remove" takes e; without it, the word is "away"
        transitive_only = top_verb in b.transitive - b.intransitive
        predicate = "adjective" if transitive_only and roles.get("adjective") else "verb"
    elif drill and drill[0][0] == "preposition":
        predicate = "adjective" if roles.get("adjective") else "verb"   # "lon." = "That's true."
    elif drill:
        predicate = drill[0][0]
    else:
        predicate = max(("adjective", "verb", "noun"), key=lambda r: (b.weights[r], -_TIE_ORDER.index(r)))
    senses = Senses(word, roles, predicate)
    for key, value in _OVERRIDES.get(word, {}).items():
        if key == "predicate":
            senses.predicate = value
        else:
            senses.roles[key] = value + [o for o in senses.roles.get(key, []) if o not in value]
    return senses


def _table(name: str) -> list[list[str]]:
    with open(_HERE / name, encoding="utf-8", newline="") as fh:
        return [row for row in csv.reader(fh, delimiter="\t") if row and not row[0].startswith("#")]


@lru_cache(maxsize=1)
def load() -> Lexicon:
    """Build the lexicon once per process."""
    drills, lists, utterances = _wasona_glosses()
    words = {w: _senses(w, e, drills.get(w, []), lists.get(w, [])) for w, e in ENTRIES.items()}
    compounds: dict[str, dict[tuple[str, ...], str]] = {}
    for tp, en, role, *_ in _table("compounds.tsv"):
        compounds.setdefault(role, {})[tuple(tp.split())] = en
    names = {name: (en, language or en) for name, en, language, *_ in _table("names.tsv")}
    return Lexicon(words, compounds, names, utterances)

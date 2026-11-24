"""
translation/english.py
The English word forms the translator needs: plurals, present-tense and -ing
verb forms, adverbs, and a/an. The vocabulary is small and fixed (the glosses
in master_dictionary.yaml and wasona_sentences.tsv), so short rules with
exception lists are enough.

A subject's agreement is one of four strings, which pick the verb form:
  "1s"  I            am / eat
  "2"   you          are / eat
  "3s"  it, the bird is / eats
  "pl"  we, they     are / eat
"""
from __future__ import annotations

import re

_IRREGULAR_PLURALS: dict[str, str] = {
    "person": "people", "child": "children", "man": "men", "woman": "women",
    "foot": "feet", "tooth": "teeth", "mouse": "mice", "goose": "geese",
    "fish": "fish", "sheep": "sheep", "deer": "deer", "leaf": "leaves",
    "knife": "knives", "life": "lives", "wife": "wives", "wolf": "wolves",
    "half": "halves", "self": "selves", "louse": "lice", "people": "people",
}

_IRREGULAR_PRESENT: dict[str, str] = {"be": "is", "have": "has", "do": "does", "go": "goes"}

# Verbs stressed on the last syllable double its consonant before -ing
# ("beginning"); one-syllable ones are caught by the rule in ing().
_DOUBLING: frozenset[str] = frozenset({"begin", "forget", "admit", "prefer", "occur", "permit", "regret"})

_IRREGULAR_ADVERBS: dict[str, str] = {
    "good": "well", "fast": "fast", "hard": "hard", "much": "much",
    "many": "a lot", "big": "greatly", "large": "greatly", "small": "a little",
    "little": "a little", "same": "the same", "other": "differently", "fun": "for fun",
}

# Nouns that take no "a" and no plural ("good food", "much money")
_MASS: frozenset[str] = frozenset("""
    water food money knowledge wisdom air fire heat sand snow wood alcohol music love
    information energy power strength rice bread milk meat medicine clothing stuff glue
    clay dough mud dirt light work sleep death metal paper salt sugar soup juice wine
    blood smoke rain ice oil gold silver news advice help peace happiness fun goodness
    hair grass skin electricity land equipment something everything nothing anything
    speech knowledge sound noise
""".split())


def is_mass(noun: str) -> bool:
    """'water', 'good food': no 'a', no plural."""
    return noun.rpartition(" ")[2] in _MASS


# Names written in lower case, even at the start of a sentence: the language's own name.
LOWERCASE_NAMES = ("toki pona",)


def takes_article(noun: str) -> bool:
    """False for 'something', 'everything' and the like, and for names: 'Emma', 'toki pona'."""
    if noun[:1].isupper() or noun in LOWERCASE_NAMES:
        return False
    return noun.rpartition(" ")[2] not in {"something", "everything", "nothing", "anything", "someone", "everyone"}


def plural(noun: str) -> str:
    """'animal' -> 'animals', 'person' -> 'people'; only the last word changes."""
    front, _, last = noun.rpartition(" ")
    if last in _IRREGULAR_PLURALS:
        form = _IRREGULAR_PLURALS[last]
    elif re.search(r"(?:s|sh|ch|x|z)$", last):
        form = last + "es"
    elif re.search(r"[^aeiou]y$", last):
        form = last[:-1] + "ies"
    else:
        form = last + "s"
    return f"{front} {form}" if front else form


def _first_word(verb: str, change) -> str:
    """Apply change to the first word of a verb ('look at' -> 'looks at')."""
    first, _, rest = verb.partition(" ")
    form = change(first)
    return f"{form} {rest}" if rest else form


def _third_person(verb: str) -> str:
    if verb in _IRREGULAR_PRESENT:
        return _IRREGULAR_PRESENT[verb]
    if re.search(r"(?:s|sh|ch|x|z|o)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


def present(verb: str, agreement: str) -> str:
    """The present tense of verb for a subject: 'eat' -> 'eats' for '3s'."""
    if verb == "be":
        return {"1s": "am", "3s": "is"}.get(agreement, "are")
    if verb in ("can", "should", "must", "will", "may"):
        return verb
    return _first_word(verb, _third_person) if agreement == "3s" else verb


def _ing(verb: str) -> str:
    if verb == "be":
        return "being"
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith(("ee", "oe", "ye")):
        return verb[:-1] + "ing"
    if verb in _DOUBLING or re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwxy]", verb):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def ing(verb: str) -> str:
    """'eat' -> 'eating', 'make' -> 'making', 'run' -> 'running'."""
    return _first_word(verb, _ing)


def adverb(adjective: str) -> str:
    """'strange' -> 'strangely', 'good' -> 'well'."""
    if adjective in _IRREGULAR_ADVERBS:
        return _IRREGULAR_ADVERBS[adjective]
    if " " in adjective or adjective.endswith("ly"):
        return adjective
    if re.search(r"[^aeiou]y$", adjective):
        return adjective[:-1] + "ily"
    if adjective.endswith("le"):
        return adjective[:-1] + "y"
    if adjective.endswith("ic"):
        return adjective + "ally"
    return adjective + "ly"


def indefinite(phrase: str) -> str:
    """'a bird', 'an animal'."""
    vowel = re.match(r"[aeiou]", phrase, re.IGNORECASE) and not re.match(
        r"(?:uni|use|usu|one|eu)", phrase, re.IGNORECASE)
    return ("an " if vowel else "a ") + phrase


def sentence(text: str, end: str = ".") -> str:
    """Capitalise the first letter (unless a lower-case name starts it) and end with the given mark."""
    text = text.strip()
    if not text:
        return text
    if not any(re.match(rf"{name}\b", text) for name in LOWERCASE_NAMES):
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?:" else text + end

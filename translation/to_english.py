"""
translation/to_english.py
Toki Pona -> English, one sentence at a time. The validator's top-ranked parse
is turned into English construct by construct:

  subject and predicates   "waso li pona."          -> "The bird is good."
  objects with e           "mi lukin e ona."        -> "I see them."
  preverbs                 "mi wile lape."          -> "I want to sleep."
  ala                      "mi wile ala tawa."      -> "I do not want to move."
  questions                "waso li lon seme?"      -> "Where is the bird?"
                           "sina moku ala moku?"    -> "Do you eat?"
                           "moli li ike anu seme?"  -> "Death is bad, right?"
  contexts before la       "ken la ona o lape."     -> "Perhaps, they should sleep."
  requests and calls       "jan Ema o, moku!"       -> "Emma, eat!"
  pi, names, numbers       "sitelen pi tomo mute"   -> "a picture of houses"

Words come from lexicon.py through a Chooser (choose.py). Toki Pona marks no
tense, so verbs are in the present; a noun is singular with "the" unless a
number, mute or a pronoun says otherwise. Modifiers come out in reverse order,
as wasona translates them ("waso suwi lili mi" = "my adorable little bird").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass, replace

from ast_nodes import (
    Clause, CompoundSubject, ContextPhrase, DisjunctPhrase, MarkedSubject, OrdinalPhrase,
    PiPhrase, PrepPhrase, Preverb, PreverbPhrase, Sentence, SimplePhrase, UnmarkedSubject, Utterance,
    VerbPhraseIntransitive, VerbPhrasePrepositional, VerbPhraseTransitive, VerbPhraseYN,
    Vocative, Word,
)
from translation import english
from translation.choose import Chooser, FirstChoice
from translation.lexicon import Lexicon, load

# Pronouns (wasona lesson 6): subject, object and possessive forms, agreement
_PRONOUNS: dict[str, tuple[str, str, str, str]] = {
    "mi":   ("I", "me", "my", "1s"),
    "sina": ("you", "you", "your", "2"),
    "ona":  ("they", "them", "their", "pl"),
    "ni":   ("this", "this", "this", "3s"),
}
# ... and with mute or a number after them (lesson 16: "mi mute" = we)
_PLURAL_PRONOUNS: dict[str, tuple[str, str, str, str]] = {
    "mi":   ("we", "us", "our", "pl"),
    "sina": ("you all", "you all", "your", "pl"),
    "ona":  ("they", "them", "their", "pl"),
    "ni":   ("these", "these", "these", "pl"),
}

# The five prepositions (wasona lessons 8 and 15)
_PREPOSITIONS: dict[str, str] = {"tawa": "to", "lon": "in", "tan": "from", "kepeken": "using", "sama": "like"}
# A preposition as the whole predicate: "X li lon Y" -> "X is in Y"
_PREPOSITION_VERBS: dict[str, tuple[str, str]] = {
    "tawa": ("go", "to"), "lon": ("be", "in"), "tan": ("be", "from"),
    "kepeken": ("use", ""), "sama": ("be", "like"),
}
# "preposition + seme" as a question: the question word, and what stays behind
# ("X li tan seme?" -> "Where is X from?"; after a verb, "tan seme" -> "why")
_PREDICATE_QUESTIONS: dict[str, tuple[str, str]] = {
    "lon": ("where", ""), "tawa": ("where", ""), "tan": ("where", "from"),
    "kepeken": ("what", ""), "sama": ("what", "like"),
}
_ADJUNCT_QUESTIONS: dict[str, str] = {"lon": "where", "tawa": "where", "tan": "why", "kepeken": "how", "sama": "how"}

# Preverbs (wasona lesson 9, more/09-01): the English verb, and how the next
# verb attaches ("to want to ...", "to finish ...ing"; ken is a modal)
_PREVERBS: dict[str, tuple[str, str]] = {
    "wile": ("want", "to"), "ken": ("can", ""), "kama": ("begin", "to"),
    "awen": ("continue", "to"), "alasa": ("try", "to"), "lukin": ("try", "to"),
    "sona": ("know", "how to"), "open": ("begin", "to"), "pini": ("finish", "ing"),
}

_STARTERS: dict[str, str] = {"taso": "but", "kin": "also", "a": "ah", "n": "hmm", "anu": "or", "en": "and"}

# Question words a noun with seme becomes (wasona lesson 17: "jan seme" = who)
_SEME_NOUNS: dict[str, str] = {"jan": "who", "ijo": "what", "tenpo": "when", "nasin": "how", "ma": "where"}

# Number words (wasona lesson 25, more/25-01) and the shorthand letters
_NUMBER_VALUES: dict[str, int] = {"wan": 1, "tu": 2, "luka": 5, "mute": 20, "ale": 100, "ali": 100}
_LETTER_VALUES: dict[str, int] = {"W": 1, "T": 2, "L": 5, "M": 20, "A": 100}
_CARDINALS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
              "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()
_ORDINALS = "zeroth first second third fourth fifth sixth seventh eighth ninth tenth".split()

# Modifiers that grade an adjective or a verb (wasona: "lili ike" = too small)
_DEGREE_ADJECTIVE: dict[str, str] = {"mute": "very", "lili": "somewhat", "wawa": "really", "ike": "too"}
_DEGREE_VERB: dict[str, str] = {"mute": "a lot", "lili": "a little", "taso": "only", "kin": "too"}

_MODALS = frozenset({"can", "should", "must", "will", "may"})

# An object that is also the subject: "mi lukin e mi" = I see myself
_REFLEXIVES: dict[str, tuple[str, str]] = {
    "I": ("me", "myself"), "you": ("you", "yourself"), "we": ("us", "ourselves"),
    "they": ("them", "themselves"), "you all": ("you all", "yourselves"),
}

# Head nouns that only say what kind of thing a name names ("jan Ema" = Emma)
_NAME_KINDS = frozenset({"jan", "ma", "lipu", "musi", "soweli", "waso", "kala", "akesi", "pipi",
                         "kulupu", "tomo", "esun", "nasin", "sitelen", "toki", "nimi"})


@dataclass
class Noun:
    """A noun phrase in English."""
    text: str
    agreement: str = "3s"         # how a verb agrees with it (english.py)
    question: str | None = None   # set when it asks: "what", "who", "which bird"
    named: bool = False           # a person's name: "jan Maja o ..." calls Maya


@dataclass
class Verb:
    """One English predicate: a verb in its base form and what follows it."""
    verb: str                     # "eat", "be", "can", "want"
    rest: str = ""                # "the fruit", "good", "to sleep", "in the house"
    negated: bool = False
    question: str | None = None   # a question word from inside it: "what", "where"
    yes_no: bool = False          # V ala V: "Do you eat?"


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def _list(items: list[str], conjunction: str = "and") -> str:
    """'X', 'X and Y', 'X, Y and Z'."""
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}" if len(items) > 1 else "".join(items)


def _words(node) -> list[Word]:
    """Every Word in a node, in sentence order."""
    found: list[Word] = []

    def walk(n) -> None:
        if isinstance(n, Word):
            found.append(n)
        elif isinstance(n, list):
            for item in n:
                walk(item)
        elif is_dataclass(n):
            for f in fields(n):
                walk(getattr(n, f.name))

    walk(node)
    return sorted(found, key=lambda w: w.offset)


def _is_name(word: Word) -> bool:
    return word.token_type == "MOD" and word.text[:1].isupper()


def _is_number(word: Word) -> bool:
    return word.token_type in ("NUM", "NUM_ABBR") or word.text in _NUMBER_VALUES


def _number(words: list[Word]) -> int:
    """Add the values up (wasona more/25-01). ale is 100, but after other number
    words it multiplies the number so far by 100, as in the 2021 system:
    "mute ale mute luka" = 2025, while "ale ale ale" = 300 as before 2021."""
    values = []
    for w in words:
        if w.text.startswith("#"):
            values += [_LETTER_VALUES[c] for c in w.text[1:]]
        else:
            values.append(_NUMBER_VALUES[w.text])
    total = group = 0
    digits = False   # a number word other than ale since the last multiplying ale
    for v in values:
        if v == 100 and digits:
            total, group, digits = (total + group) * 100, 0, False
        else:
            group += v
            digits = digits or v != 100
    return total + group


def _cardinal(n: int) -> str:
    return _CARDINALS[n] if n < len(_CARDINALS) else str(n)


def _ordinal(n: int) -> str:
    if n < len(_ORDINALS):
        return _ORDINALS[n]
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# Verb forms
# ---------------------------------------------------------------------------

def _finite(v: Verb, agreement: str) -> str:
    """'eats the fruit', 'does not eat', 'is good', 'cannot sleep'."""
    if v.verb == "be" or v.verb in _MODALS:
        verb = english.present(v.verb, agreement)
        if v.negated:
            verb = "cannot" if v.verb == "can" else f"{verb} not"
        return _join(verb, v.rest)
    if v.negated:
        return _join(english.present("do", agreement), "not", v.verb, v.rest)
    return _join(english.present(v.verb, agreement), v.rest)


def _base(v: Verb) -> str:
    """Without tense: 'eat the fruit', 'not eat', 'be good'."""
    return _join("not" if v.negated else "", v.verb, v.rest)


def _inverted(v: Verb, subject: Noun) -> str:
    """Question order: 'is the bird good', 'does the bird eat', 'can you sleep'."""
    neg = "not" if v.negated else ""
    if v.verb == "be" or v.verb in _MODALS:
        return _join(english.present(v.verb, subject.agreement), subject.text, neg, v.rest)
    return _join(english.present("do", subject.agreement), subject.text, neg, v.verb, v.rest)


def _imperative(v: Verb) -> str:
    """'eat the fruit', 'do not eat', 'be good'."""
    return _join("do not", v.verb, v.rest) if v.negated else _join(v.verb, v.rest)


def _and(first: Verb, others: list[Verb], form) -> str:
    """Further predicates after the first: 'and sleeps', 'and big'."""
    out = []
    for v in others:
        out.append(v.rest if v.verb == first.verb == "be" and not v.negated else form(v))
    return _join(*[f"and {o}" for o in out])


def _realise(subject: Noun | None, verbs: list[Verb], request: bool) -> tuple[str, bool]:
    """English for a clause, and whether it asks a question."""
    first, others = verbs[0], verbs[1:]
    if request:
        # "o moku!" and "sina o moku!" ask the listener, "jan Maja o pana ..."
        # calls Maya first (wasona lesson 12); anyone else should
        if subject is None or subject.text in ("you", "you all"):
            return _join(_imperative(first), _and(first, others, _imperative)), False
        if subject.named:
            return f"{subject.text}, " + _join(_imperative(first), _and(first, others, _imperative)), False
        verbs = [Verb("should", _base(v), question=v.question, yes_no=v.yes_no) for v in verbs]
        first, others = verbs[0], verbs[1:]
    asked = next((v.question for v in verbs if v.question), None)
    if subject.question:   # "seme li moku?" -> "What eats?"
        return _join(subject.question, _finite(first, "3s"), _and(first, others, _base)), True
    if asked:              # "sina moku e seme?" -> "What do you eat?"
        return _join(asked, _inverted(first, subject), _and(first, others, _base)), True
    if any(v.yes_no for v in verbs):
        return _join(_inverted(first, subject), _and(first, others, _base)), True
    return _join(subject.text, _finite(first, subject.agreement),
                 _and(first, others, lambda v: _finite(v, subject.agreement))), False


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

class EnglishRenderer:
    """Turns parsed Toki Pona into English; one instance can render many sentences."""

    def __init__(self, lexicon: Lexicon | None = None, chooser: Chooser | None = None):
        self.lexicon = lexicon or load()
        self.chooser = chooser or FirstChoice()
        self.source = ""   # the Toki Pona being rendered, for the chooser

    # --- words -------------------------------------------------------------

    def _word(self, word: str, role: str, *fallback: str) -> str | None:
        """The chosen English for word in role, trying the fallback roles next."""
        senses = self.lexicon.get(word)
        for r in (role, *fallback):
            options = senses.options(r) if senses else []
            if options:
                return self.chooser.rank(word, r, options, self.source)[0][0]
        return None

    def _compound(self, words: list[str], role: str) -> tuple[str, int] | None:
        """The longest compound the words start with: (English, words it takes)."""
        table = self.lexicon.compounds.get(role, {})
        for n in range(len(words), 1, -1):
            if tuple(words[:n]) in table:
                return table[tuple(words[:n])], n
        return None

    def _modifier(self, word: str) -> tuple[str, str]:
        """('adjective' | 'noun', English) for a content word used as a modifier."""
        senses = self.lexicon.get(word)
        if senses and senses.predicate == "noun":
            noun = self._word(word, "noun")
            if noun:
                return "noun", noun        # "tomo telo" = water building
        adjective = self._word(word, "adjective")
        if adjective:
            return "adjective", adjective
        verb = self._word(word, "verb")
        if verb:
            return "adjective", english.ing(verb)   # "jan toki" = talking person
        return "noun", self._word(word, "noun") or word

    def _name(self, kind: str, name: str) -> str:
        """'Germany' for "ma Tosi", 'German' for "toki Tosi"."""
        place, language = self.lexicon.names.get(name, (name, name))
        if kind == "toki":
            return language
        if kind == "sitelen":
            return f"{language} writing"
        if kind == "nimi":
            return f"the word '{name.lower()}'"
        return place

    # --- noun phrases ------------------------------------------------------

    def noun(self, node, article: str = "the", case: str = "subject",
             described: list[str] | None = None) -> Noun:
        """A noun phrase. article is "the", "a" or "" (none); case is "subject"
        or "object"; described adds adjective phrases from pi groups."""
        match node:
            case UnmarkedSubject():
                return self._pronoun(node.word.text, [], case)
            case MarkedSubject():
                return self.noun(node.phrase, article, case)
            case CompoundSubject() | DisjunctPhrase():
                parts = [self.noun(p, article, case) for p in getattr(node, "parts", None) or node.options]
                joiner = "and" if getattr(node, "conjunction", "anu") == "en" else "or"
                agreement = "pl" if joiner == "and" else parts[-1].agreement
                question = next((p.question for p in parts if p.question), None)
                return Noun(_list([p.text for p in parts], joiner), agreement, question)
            case SimplePhrase():
                return self._simple(node.head, node.modifiers, article, case, described or [])
            case OrdinalPhrase():
                n = _number(node.ordinal.words) if node.ordinal.words else 1
                base = node.base
                return self._simple(base.head, base.modifiers, "", case, described or [],
                                    determiner=f"the {_ordinal(n)}")
            case PiPhrase():
                return self._pi(node, article, case)
        return Noun(" ".join(w.text for w in _words(node)))

    def _pi(self, node: PiPhrase, article: str, case: str) -> Noun:
        """pi groups: adjective groups describe the head ("soweli pi suwi mute" =
        a very lovely cat); noun groups follow with "of" (lesson 22)."""
        adjectives, nouns = [], []
        for group in node.pi_groups:
            base = group.base if isinstance(group, OrdinalPhrase) else group
            senses = self.lexicon.get(base.head.text)
            if (isinstance(group, SimplePhrase) and senses and senses.predicate == "adjective"
                    and not any(_is_name(m) for m in group.modifiers)):
                adjectives.append(self._adjective(group.head, group.modifiers)[0])
            else:
                nouns.append(self.noun(group, "the", "object"))
        head = self.noun(node.base, article, case, described=adjectives)
        question = head.question or next((n.question for n in nouns if n.question), None)
        text = _join(head.text, *[f"of {n.text}" for n in nouns])
        return Noun(text, head.agreement, question)

    def _pronoun(self, word: str, mods: list[Word], case: str) -> Noun:
        plural = bool(mods) and (mods[0].text == "mute" or mods[0].token_type in ("NUM", "NUM_ABBR"))
        subject, obj, _, agreement = (_PLURAL_PRONOUNS if plural else _PRONOUNS)[word]
        text = subject if case == "subject" else obj
        for m in mods[1:] if plural else mods:
            if m.text == "taso":
                text = f"only {text}"
            elif m.text == "kin":
                text = f"{text} too"
            elif m.text in ("ale", "ali"):
                text = f"all of {obj}"
        return Noun(text, agreement)

    def _simple(self, head: Word, mods: list[Word], article: str, case: str,
                described: list[str], determiner: str | None = None) -> Noun:
        mods = [m for m in mods if m.token_type != "A"]   # emphasis
        if head.text in _PRONOUNS:
            return self._pronoun(head.text, mods, case)
        if head.text == "seme":
            return Noun("what", question="what")
        if _is_number(head) and mods and all(_is_number(m) for m in mods):
            return Noun(_cardinal(_number([head, *mods])))   # "ale ale ale" = 300
        if head.token_type == "PREP" and mods and not described:
            # a preposition the parser read as a phrase: "tawa jan Ali" = to Ali
            rest = self._simple(mods[0], mods[1:], "the", "object", [])
            return Noun(_join(_PREPOSITIONS.get(head.text, head.text), rest.text), question=rest.question)
        if mods and mods[-1].text == "seme" and head.text in _SEME_NOUNS and len(mods) == 1:
            word = _SEME_NOUNS[head.text]
            return Noun(word, question=word)
        names = [i for i, m in enumerate(mods) if _is_name(m)]
        if names:
            return self._named(head, mods, names[0], article, case)
        found = self._compound([head.text] + [m.text for m in mods], "noun")
        if found:
            noun, used = found
            mods = mods[used - 1:]
        elif verb := self._compound([head.text] + [m.text for m in mods], "verb"):
            noun, mods = english.ing(verb[0]), mods[verb[1] - 1:]   # "kama sona" = learning
        else:
            noun = self._word(head.text, "noun", "adjective", "verb") or head.text
        return self._describe(noun, mods, article, described, determiner)

    def _named(self, head: Word, mods: list[Word], at: int, article: str, case: str) -> Noun:
        """A head noun and a name: "jan Ema" = Emma, "toki Inli" = English. With
        words between them, the name describes them: "soweli jan Ema" = a
        human-like animal named Emma (wasona lesson 23)."""
        name = " ".join(self._name(head.text, m.text) for m in mods[at:] if _is_name(m))
        kind = (head.text, *[m.text for m in mods[:at]])
        if (at == 0 and head.text in _NAME_KINDS) or kind in self.lexicon.compounds.get("noun", {}):
            return Noun(name, named=head.text == "jan")   # "jan Ema" = Emma, "ma tomo Towano" = Toronto
        kind = self._simple(head, mods[:at], article, case, [])
        return Noun(f"{kind.text} named {name}", kind.agreement)

    def _describe(self, noun: str, mods: list[Word], article: str,
                  described: list[str], determiner: str | None = None) -> Noun:
        """Put a noun's modifiers in English order: determiner or number, then
        adjectives, then nouns, then the noun ("tomo telo suli mi" = my big water building)."""
        adjectives: list[str] = list(described)
        adjuncts: list[str] = []
        count: str | None = None
        plural = only = also = False
        question = None
        i = 0
        while i < len(mods):
            m, w = mods[i], mods[i].text
            if w in _PRONOUNS:
                plural_owner = i + 1 < len(mods) and mods[i + 1].text == "mute" and w != "ni"
                determiner = (_PLURAL_PRONOUNS if plural_owner else _PRONOUNS)[w][2]
                if w == "ni":
                    determiner = "this"
                i += 2 if plural_owner else 1
                continue
            if _is_number(m):
                run = [m]
                while i + len(run) < len(mods) and _is_number(mods[i + len(run)]):
                    run.append(mods[i + len(run)])
                i += len(run)
                if len(run) == 1 and w == "mute":
                    count, plural = "many", True        # "soweli mute" = many cats
                elif len(run) == 1 and w in ("ale", "ali"):
                    count, plural = "all", True
                else:
                    n = _number(run)
                    count, plural = _cardinal(n), n != 1
                continue
            i += 1
            if w == "ala":
                determiner = "no"
            elif w == "seme":
                determiner, question = "which", True
            elif w == "taso":
                only = True
            elif w == "kin":
                also = True
            else:
                kind, text = self._modifier(w)
                (adjuncts if kind == "noun" else adjectives).append(text)
        mass = english.is_mass(noun)   # "much money", "good food"
        if mass and plural:
            plural, count = False, "much" if count == "many" else count
        head = english.plural(noun) if plural else noun
        words = _join(*reversed(adjectives), *reversed(adjuncts), head)
        if count:
            words = _join(count, words)
        if determiner:
            words = _join(determiner, words)
        elif not english.takes_article(noun) or noun.startswith("the "):
            pass   # "ijo" = something; "tenpo kama" = the future
        elif not count and article == "the":
            words = f"the {words}"
        elif not count and article == "a" and not plural and not mass:
            words = english.indefinite(words)
        if only:
            words = f"only {words}"
        if also:
            words = f"{words} too"
        text = words
        return Noun(text, "pl" if plural else "3s", text if question else None)

    def _adjective(self, head: Word, mods: list[Word]) -> tuple[str, bool]:
        """An adjective and its modifiers ("pona mute" = very good), and whether ala negates it."""
        adjective = self._word(head.text, "adjective", "noun", "verb") or head.text
        degrees, negated = [], False
        for m in mods:
            if m.text == "ala":
                negated = True
            elif m.text in _DEGREE_ADJECTIVE:
                degrees.append(_DEGREE_ADJECTIVE[m.text])
            elif m.token_type != "A" and not _is_name(m):
                degrees.append(english.adverb(self._word(m.text, "adjective", "noun") or m.text))
        return _join(*reversed(degrees), adjective), negated

    # --- predicates --------------------------------------------------------

    def _phrase_parts(self, phrase) -> tuple[Word, list[Word]]:
        """The head and modifiers of a verb's phrase (pi groups are left out)."""
        base = phrase.base if isinstance(phrase, (PiPhrase, OrdinalPhrase)) else phrase
        if isinstance(base, OrdinalPhrase):
            base = base.base
        return base.head, list(base.modifiers)

    def _preps(self, preps: list[PrepPhrase]) -> tuple[list[str], str | None]:
        """Prepositional phrases after a verb, and a question word from one with seme."""
        out, question = [], None
        for pp in preps:
            word = pp.prep.word.text
            obj = self.noun(pp.obj, "the", "object")
            if obj.question == "what":
                question = question or _ADJUNCT_QUESTIONS.get(word, "what")
            elif obj.question:
                question = question or _join(_PREPOSITIONS.get(word, word), obj.question)
            else:
                out.append(_join("not" if pp.prep.negated else "", _PREPOSITIONS.get(word, word), obj.text))
        return out, question

    def _verb_mods(self, head: Word, mods: list[Word]) -> tuple[list[str], list[str], bool, str | None]:
        """A verb's modifiers: (objects from names, adverbs, negated, question word)."""
        objects, adverbs, negated, question = [], [], False, None
        numbers = [m for m in mods if _is_number(m) and m.text != "mute"]
        if numbers:   # "toki tu wan": a count, said once
            adverbs.append(_cardinal(_number(numbers)))
        for m in mods:
            if m in numbers:
                continue
            if m.text == "ala":
                negated = True
            elif m.text == "seme":
                question = "what"
            elif _is_name(m):
                objects.append(self._name(head.text, m.text))   # "toki Inli" = speak English
            elif m.text in _DEGREE_VERB:
                adverbs.append(_DEGREE_VERB[m.text])
            elif m.token_type != "A" and m.text not in _PRONOUNS:
                adverbs.append(english.adverb(self._word(m.text, "adjective", "noun") or m.text))
        return objects, adverbs, negated, question

    def _verb(self, head: Word, mods: list[Word]) -> tuple[str, str, list[Word]]:
        """The English verb for a head and its modifiers: (verb, what follows its
        object, the modifiers a compound didn't use)."""
        found = self._compound([head.text] + [m.text for m in mods], "verb")
        if found:
            return found[0], "", mods[found[1] - 1:]
        if head.text in ("toki", "musi") and any(_is_name(m) for m in mods):
            return {"toki": "speak", "musi": "play"}[head.text], "", mods   # "toki Kanse" = speak French
        verb = self._word(head.text, "verb")
        if verb and re.fullmatch(r"make (?!a |an |the )\w+", verb):
            return "make", verb[5:], mods   # "pona e lukin" = make the eye better
        if verb:
            return verb, "", mods
        # wasona lesson 5: any word can be a verb ("waso" = to make into a bird)
        adjective = self._word(head.text, "adjective")
        if adjective:
            return "make", adjective, mods
        noun = self._word(head.text, "noun") or head.text
        return "turn", f"into {noun if english.is_mass(noun) else english.indefinite(noun)}", mods

    def _transitive(self, phrase, objects, preps) -> Verb:
        head, mods = self._phrase_parts(phrase)
        verb, complement, mods = self._verb(head, mods)
        if re.search(r" (?:a|an|the) ", f" {verb} ") and objects:
            # "make a sound" has its own object; with e, take a verb without one
            senses = self.lexicon.get(head.text)
            options = [o for o in (senses.options("verb") if senses else []) if not re.search(r"\b(?:a|an|the)\b", o)]
            verb = options[0] if options else verb
        if head.text == "toki" and verb in ("talk", "speak", "say"):
            # "toki e ni" = say this; "toki e tomo" = talk about a house (more/08-02)
            says = all([w.text for w in _words(o.np)] == ["ni"] for o in objects)
            verb = "say" if says else "talk about"
        named, adverbs, negated, question = self._verb_mods(head, mods)
        texts = []
        for o in objects:
            obj = self.noun(o.np, "the", "object")
            if obj.question:
                question = question or obj.question
                continue
            obj_preps, q = self._preps(o.preps)
            question = question or q
            texts.append(_join(obj.text, *obj_preps))
        after, q = self._preps(preps)
        rest = _join(*named, _list(texts), complement, *adverbs, *after)
        return Verb(verb, rest, negated, question or q)

    def _intransitive(self, phrase, preps) -> Verb:
        head, mods = self._phrase_parts(phrase)
        after, question = self._preps(preps)
        if head.text == "seme":   # "nimi sina li seme?" = What is your name?
            return Verb("be", _join(*after), question="what")
        senses = self.lexicon.get(head.text)
        role = senses.predicate if senses else "noun"
        named = any(_is_name(m) for m in mods)
        if self._compound([head.text] + [m.text for m in mods], "verb") or (named and head.text in ("toki", "musi")):
            role = "verb"   # "jan li musi Manka." = The person plays Minecraft (more/08-02)
        if head.text in _PRONOUNS or (named and head.text not in ("toki", "musi")):
            role = "noun"   # "ni li sitelen Lasina" = this is Latin writing; but "toki Inli" = speak English
        if role == "verb":
            verb, complement, mods = self._verb(head, mods)
            named, adverbs, negated, q = self._verb_mods(head, mods)
            return Verb(verb, _join(*named, complement, *adverbs, *after), negated, question or q)
        if role == "adjective" and any(m.text in _PRONOUNS for m in mods):
            role = "noun"   # a pronoun makes it possessive: "pona tawa mi" = my moving goodness, not "mily good"
        if role == "adjective":
            adjective, negated = self._adjective(head, mods)
            return Verb("be", _join(adjective, *after), negated, question)
        negated = any(m.text == "ala" for m in mods)
        kept = [m for m in mods if m.text != "ala"]
        node = replace(phrase, modifiers=kept) if isinstance(phrase, SimplePhrase) else phrase
        noun = self.noun(node, "a", "object")
        return Verb("be", _join(noun.text, *after), negated, question or noun.question)

    def _prepositional(self, preps: list[PrepPhrase]) -> Verb:
        first, others = preps[0], preps[1:]
        word = first.prep.word.text
        verb, prep = _PREPOSITION_VERBS.get(word, ("be", word))
        obj = self.noun(first.obj, "the", "object")
        after, question = self._preps(others)
        if obj.question == "what":
            question, stays = _PREDICATE_QUESTIONS.get(word, ("what", prep))
            rest = _join(stays, *after)
        elif obj.question:
            question, rest = _join(prep, obj.question), _join(*after)
        else:
            rest = _join(prep, obj.text, *after)
        return Verb(verb, rest, first.prep.negated, question, yes_no=first.prep.question)

    def _preverb(self, word: str, negated: bool, asks: bool, inner: Verb) -> Verb:
        if word in ("kama", "awen") and inner.verb == "be":   # "kama suli" = become big, "awen lon" = stay in
            verb = "become" if word == "kama" else "stay"
            return Verb(verb, inner.rest, negated, inner.question, asks or inner.yes_no)
        verb, link = _PREVERBS.get(word, ("try", "to"))
        if link == "ing":
            complement = _join("not" if inner.negated else "", english.ing(inner.verb), inner.rest)
        else:
            complement = _join(link, _base(inner))
        return Verb(verb, complement, negated, inner.question, asks or inner.yes_no)

    def predicate(self, node) -> Verb:
        match node:
            case PreverbPhrase():
                preverbs = list(node.preverbs)
                verb = self.predicate(node.vp)
                inner = getattr(node.vp, "verb", None)
                if inner is not None:
                    pair = (preverbs[-1].word.text, self._phrase_parts(inner)[0].text)
                    compound = self.lexicon.compounds.get("verb", {}).get(pair)
                    if compound:   # "kama sona" = learn
                        last = preverbs.pop()
                        verb = Verb(compound, verb.rest if verb.verb != "be" else "",
                                    verb.negated or last.negated, verb.question, verb.yes_no or last.question)
                for pv in reversed(preverbs):
                    verb = self._preverb(pv.word.text, pv.negated, pv.question, verb)
                return verb
            case VerbPhraseTransitive():
                return self._transitive(node.verb, node.objects, node.preps)
            case VerbPhraseIntransitive():
                return self._intransitive(node.verb, node.preps)
            case VerbPhrasePrepositional():
                return self._prepositional(node.preps)
            case VerbPhraseYN():
                verb = (self._transitive(node.verb, node.objects, node.preps) if node.objects
                        else self._intransitive(node.verb, node.preps))
                return replace(verb, yes_no=True)
        raise TypeError(f"not a predicate: {type(node).__name__}")

    # --- clauses and sentences ---------------------------------------------

    def _fragment(self, subject) -> str:
        """A phrase with no verb: an answer, a greeting, a label (lesson 18)."""
        phrase = subject.phrase if isinstance(subject, MarkedSubject) else subject
        if isinstance(phrase, SimplePhrase):
            mods = [m for m in phrase.modifiers if m.token_type != "A"]
            if any(m.text == "ala" for m in mods):   # "musi ala." = not fun
                kept = [m for m in mods if m.text != "ala"]
                return _join("not", self._fragment(replace(phrase, modifiers=kept)))
            if phrase.head.token_type == "PREV" and mods:
                inner = self.lexicon.get(mods[0].text)
                pair = (phrase.head.text, mods[0].text)
                if (inner and inner.predicate == "verb") or pair in self.lexicon.compounds.get("verb", {}):
                    rest = SimplePhrase(mods[0], mods[1:], "")
                    verb = self.predicate(PreverbPhrase([_preverb_of(phrase.head)], VerbPhraseIntransitive(rest, [], ""), ""))
                    return f"to {_base(verb)}"   # "wile lape" = to want to sleep
            senses = self.lexicon.get(phrase.head.text)
            if not mods and senses and phrase.head.text not in _PRONOUNS:
                role = senses.predicate
                return self._word(phrase.head.text, role, "noun", "adjective", "verb") or phrase.head.text
        return self.noun(subject, "a", "subject").text

    def _likes(self, node: Clause) -> tuple[str, bool] | None:
        """wasona's "kasi li pona tawa mi." = I like plants (Plants are good to me)."""
        if node.predicate_type == "o" or node.subject is None or len(node.predicates) != 1:
            return None
        p = node.predicates[0]
        if not isinstance(p, VerbPhraseIntransitive) or len(p.preps) != 1:
            return None
        head, mods = self._phrase_parts(p.verb)
        if head.text not in ("pona", "ike") or p.preps[0].prep.word.text != "tawa" \
                or any(m.text != "ala" for m in mods):
            return None
        liker = self.noun(p.preps[0].obj, "the", "subject")
        liked = self.noun(node.subject, "the", "object")
        verb = Verb("like" if head.text == "pona" else "dislike", "" if liked.question else liked.text,
                    negated=bool(mods), question=liked.question)
        return _realise(liker, [verb], False)

    def clause(self, node: Clause) -> tuple[str, bool]:
        """English for a clause, and whether it asks a question."""
        if not node.predicates:
            fragment = self._fragment(node.subject)
            return fragment, fragment in ("what", "who") or "which" in fragment.split()
        likes = self._likes(node)
        if likes:
            return likes
        subject = self.noun(node.subject, "the", "subject") if node.subject else None
        verbs = [self.predicate(p) for p in node.predicates]
        if subject is None and node.predicate_type != "o":
            subject = Noun("it")
        if subject is not None and subject.text in _REFLEXIVES:
            plain, reflexive = _REFLEXIVES[subject.text]
            verbs = [replace(v, rest=re.sub(rf"^{plain}\b", reflexive, v.rest)) for v in verbs]
        return _realise(subject, verbs, node.predicate_type == "o")

    def _context(self, ctx: ContextPhrase) -> str:
        content = ctx.content
        if isinstance(content, Clause):
            text, _ = self.clause(content)
            return f"when {text}"   # wasona: "suno li pini la, mi lape." = When the sun sets, I sleep.
        words = [w.text for w in _words(content)]
        fixed = self.lexicon.compounds.get("context", {}).get(tuple(words))
        if fixed:
            return fixed
        if words == ["mi"]:
            return "for me"
        senses = self.lexicon.get(words[0]) if len(words) == 1 else None
        if senses and senses.predicate == "adjective":   # "lete la" = when it is cold
            return f"when it is {self._word(words[0], 'adjective') or words[0]}"
        return f"as for {self.noun(content, 'the', 'object').text}"   # lesson 20: "as for ..."

    def sentence(self, node: Sentence) -> tuple[str, bool]:
        main, asks = self.clause(node.clause)
        if node.anu_seme:   # wasona lesson 17: "..., right?"
            main, asks = f"{main}, right", True
        parts = [self._context(c) + "," for c in node.contexts]
        starters = [_STARTERS.get(w.text, w.text) for s in node.starters for w in s.words]
        text = _join(*starters, *parts, main)
        if node.vocative is not None:
            name = self.noun(node.vocative.addressee, "", "subject").text
            before = _words(node.vocative)[0].offset < _words(node.clause)[0].offset if _words(node.clause) else True
            text = f"{name}, {text}" if before else f"{text}, {name}"
        if node.elaboration is not None:
            more, more_asks = self.sentence(node.elaboration)
            text, asks = f"{text}: {more}", asks or more_asks
        return text, asks

    def utterance(self, utt: Utterance, source: str) -> str:
        """English for one parsed sentence; source is its Toki Pona text."""
        self.source = source
        plain = source.strip().lower()
        if plain in self.lexicon.utterances:   # "toki!" = Hello! Let's talk!
            return self.lexicon.utterances[plain]
        end = next((mark for mark in "?!:" if source.rstrip().endswith(mark)), ".")
        words = tuple(re.findall(r"[a-zA-Z]+", source))
        if words in self.lexicon.compounds.get("utterance", {}):
            return english.sentence(self.lexicon.compounds["utterance"][words], end)
        starters = [_STARTERS.get(w.text, w.text) for s in utt.starters for w in s.words]
        body = utt.body
        if body is None:
            return english.sentence(" ".join(starters), end)
        if isinstance(body, Vocative):
            return english.sentence(_join(*starters, self.noun(body.addressee, "", "subject").text), "!")
        text, asks = self.sentence(body)
        return english.sentence(_join(*starters, text), "?" if asks else end)

    def gloss(self, source: str) -> str:
        """Word by word, for a sentence that doesn't parse."""
        words = [self._word(w, "noun", "verb", "adjective") or w for w in re.findall(r"[a-zA-Z]+", source)]
        return english.sentence(" ".join(words))


def _preverb_of(word: Word) -> Preverb:
    """A plain Preverb node for a word."""
    return Preverb(word, negated=False, emphasis=False, source=word.text)


@dataclass
class Translation:
    """One sentence and its English."""
    toki_pona: str
    english: str
    parsed: bool   # False: no parse, so the English is a word-by-word gloss


def translate(text: str, renderer: EnglishRenderer | None = None) -> list[Translation]:
    """English for each sentence of text."""
    from validator import validate_text
    renderer = renderer or EnglishRenderer()
    out = []
    for result in validate_text(text):
        if result.utterance is None:
            out.append(Translation(result.text, renderer.gloss(result.text), False))
        else:
            out.append(Translation(result.text, renderer.utterance(result.utterance, result.text), True))
    return out

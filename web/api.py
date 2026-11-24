"""
web/api.py
The web page's one entry point. web/worker.js runs it inside Pyodide:

    analyse(text) -> JSON: one object per sentence, with each of its readings
                     (English, the validator's notes and the parse as a
                     tree, likeliest first), and the dictionary entry when
                     the sentence is a single word

web/build.py also uses entry() to write the dictionary page. The same
validation feeds both the notes and the English, so each sentence is parsed
once.
"""
from __future__ import annotations

import json
import re
from dataclasses import fields, is_dataclass
from pathlib import Path

from ast_nodes import Word
from lexer import ENTRIES, tokenize
from translation.to_english import EnglishRenderer
from validator import diagnose, validate_text

# What each construct is called on the page
_KINDS = {
    "Utterance": "Utterance",
    "Sentence": "Sentence",
    "SentenceStarter": "Starter",
    "Clause": "Clause",
    "MarkedSubject": "Subject",
    "UnmarkedSubject": "Subject",
    "CompoundSubject": "Subjects",
    "SimplePhrase": "Phrase",
    "PiPhrase": "Phrase with pi",
    "OrdinalPhrase": "Ordinal",
    "DisjunctPhrase": "Choice (anu)",
    "Number": "Number",
    "VerbPhraseTransitive": "Verb and object",
    "VerbPhraseIntransitive": "Verb",
    "VerbPhrasePrepositional": "Preposition as verb",
    "VerbPhraseYN": "Yes-no question",
    "PreverbPhrase": "Preverb and verb",
    "Preverb": "Preverb",
    "ObjectPhrase": "Object (e)",
    "PrepPhrase": "Prepositional phrase",
    "Preposition": "Preposition",
    "ContextPhrase": "Context (la)",
    "Vocative": "Call (o)",
    "Word": "Word",
}
# The role a field gives its value ("" says nothing new)
_ROLES = {
    "body": "", "clause": "", "phrase": "", "content": "", "vp": "", "base": "",
    "subject": "subject", "predicates": "predicate", "objects": "object", "np": "",
    "preps": "", "prep": "", "obj": "object of the preposition", "head": "head",
    "modifiers": "modifier", "verb": "", "repeated_verb": "repeated verb",
    "preverbs": "", "contexts": "", "starters": "", "words": "", "word": "",
    "pi_groups": "after pi", "options": "option", "parts": "", "addressee": "",
    "vocative": "", "elaboration": "after the colon", "ordinal": "",
}


def _tree(node: object, role: str = "") -> dict:
    """A construct as {kind, role, text, children}, with chains that repeat the same words folded into one line."""
    kind = _KINDS.get(type(node).__name__, type(node).__name__)
    if isinstance(node, Word):
        return {"kind": kind, "role": role, "text": node.text, "children": []}
    children = []
    for f in fields(node):
        value = getattr(node, f.name)
        child_role = _ROLES.get(f.name, f.name.replace("_", " "))
        for item in value if isinstance(value, list) else [value]:
            if is_dataclass(item):
                children.append(_tree(item, child_role))
    text = getattr(node, "source", "")
    # "Subject" over a "Phrase" with the same words reads as one line, "Subject · Phrase", and a phrase of one
    # word needs no line for the word.
    while len(children) == 1 and children[0]["text"] == text:
        only = children[0]
        if only["kind"] not in (kind, "Word"):
            kind = f"{kind} · {only['kind']}"
        role = role or only["role"]
        children = only["children"]
    if role == kind.split(" · ")[0].lower():
        role = ""
    return {"kind": kind, "role": role, "text": text, "children": children}


# The parts of speech an entry groups its meanings under, in the order the page lists them: the name, and a line
# with an English example or two for someone who hasn't met the term.
PARTS_OF_SPEECH = {
    "n": ("noun", "A person, place, thing, or idea, like “dog” in “The dog sleeps” or “peace” in “We want peace.”"),
    "pronoun": ("pronoun", "A word that stands in for a noun, like “she” in “She reads” or “they” in “They left.”"),
    "m": ("modifier", "A word that describes another, like “big” in “a big house” or “quickly” in “run quickly.” "
                      "In toki pona it comes after the word it describes."),
    "v": ("verb", "An action or a state, like “run” in “Dogs run.” Some verbs act on something, like “eat” in “I "
                  "eat the apple” (in toki pona that thing comes after e); others stand alone, like “sleep” in “The "
                  "baby sleeps.”"),
    "vt": ("with an object", ""),
    "vi": ("without an object", ""),
    "a": ("preverb", "A verb that comes before another verb, like “can” in “I can swim” or “want” in “I want to eat.”"),
    "p": ("preposition", "A word that places something, like “in” in “in the house” or “with” in “with my friend.”"),
    "c": ("conjunction", "A word that joins ideas, like “or” in “Tea or coffee?” or “but” in “I’m tired but happy.”"),
    "f": ("particle", "A small grammar word that shows how a sentence fits together rather than naming anything, like "
                      "“the” in “the dog.” toki pona’s li, which comes before the verb, is one."),
    "#": ("number", "A count, like “one” in “one cat” or “two” in “two birds.”"),
    "i": ("interjection", "A word said on its own as an exclamation, like “Wow!” or “Thanks!”"),
    "la": ("as context, before la", "The meaning when the word sets the scene before la, the way “If…” or “When…” does in "
                        "“In the morning, I eat.”"),
    "kama": ("after kama", "The meaning after kama (to come, to become), like “get” in “get tired” or “come to” in "
                           "“come to know.”"),
    "other": ("other", "Uses that don’t fit the parts of speech above."),
}
# lipu Linku's part-of-speech tags, as the tags above
_LINKU_TAGS = {
    "particle": "f", "preverb": "a", "number": "#", "preposition": "p", "interjection": "i",
    "pronoun": "pronoun", "third-person pronoun": "pronoun", "adjective": "m",
}
_LINKU_SEGMENT = re.compile(r"^\(([a-z -]+)\)\s+(.+)$", re.S)


def _tidy(text: str) -> str:
    return " ".join(text.split())


def _linku_parts() -> dict[tuple[str, str], str]:
    """web/linku_parts.tsv: the part of speech of each piece of a Linku definition that doesn't name one."""
    parts = {}
    with open(Path(__file__).with_name("linku_parts.tsv"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                word, tag, piece = line.rstrip("\n").split("\t")
                parts[(word, _tidy(piece))] = tag
    return parts


LINKU_PARTS = _linku_parts()


def linku_pieces(word: str) -> list[tuple[str | None, str]]:
    """A word's Linku definition as (tag, piece): the tag it names, or the one linku_parts.tsv gives it, or None."""
    out = []
    for segment in _split_top(_tidy(ENTRIES[word].get("definition") or ""), ";"):
        tagged = _LINKU_SEGMENT.match(segment)
        if tagged and tagged[1] in _LINKU_TAGS:
            out.append((_LINKU_TAGS[tagged[1]], tagged[2]))
        else:
            out.append((LINKU_PARTS.get((word, segment)), segment))
    return out


def _split_top(text: str, separator: str) -> list[str]:
    """text split at separator, but never inside brackets or quotes."""
    pieces, depth, quoted, start = [], 0, False, 0
    for i, c in enumerate(text):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == '"':
            quoted = not quoted
        elif c == separator and depth == 0 and not quoted:
            pieces.append(text[start:i])
            start = i + 1
    pieces.append(text[start:])
    return [p.strip() for p in pieces if p.strip()]


def entry(word: str) -> dict | None:
    """A word's dictionary entry for the page: its one-line gloss, every meaning from every source grouped by part of
    speech, and the English the community most often gives it.

    Nothing is dropped: lipu Linku's definition is split at its semicolons, each piece going under the part of speech
    it names or, since Linku mixes them, the one web/linku_parts.tsv gives it. Within a part of speech the sources'
    meanings are pooled word by word, so only a meaning spelled exactly the same is listed once. Whatever wasona's
    short gloss adds joins the one-line meaning."""
    raw = ENTRIES.get(word)
    if raw is None:
        return None
    groups: dict[str, list[str]] = {}

    def add(tag: str, gloss: str) -> None:
        glosses = groups.setdefault(tag, [])
        # A gloss with brackets, quotes, or an "e.g." reads as one piece; a plain list is pooled word by word.
        whole = any(c in gloss for c in '()[]"') or "e.g." in gloss
        for piece in [gloss] if whole else _split_top(gloss, ","):
            if piece not in glosses:
                glosses.append(piece)

    for tag, piece in linku_pieces(word):
        add(tag or "other", piece)
    for sense in raw.get("senses") or []:
        add(sense.get("pos") if sense.get("pos") in PARTS_OF_SPEECH else "other", _tidy(sense["gloss"]))
    # A short meaning wrapped in brackets, like a's "(emphasis, emotion or confirmation)", is still the meaning.
    brief = _tidy(raw.get("brief", ""))
    if brief.startswith("(") and brief.endswith(")") and brief.count("(") == 1:
        brief = brief[1:-1]
    # What wasona's gloss adds joins the one-line meaning, unless the entry already says it.
    listed = {gloss for glosses in groups.values() for gloss in glosses}
    listed |= {piece for part in _split_top(brief, ";") for piece in _split_top(part, ",")}
    wasona = [p for part in _split_top(_tidy(raw.get("wasona") or ""), ";") for p in _split_top(part, ",")]
    extra = [piece for piece in wasona if piece not in listed]
    if extra:
        brief = "; ".join(filter(None, [brief, ", ".join(extra)]))
    listed = {gloss.lower() for glosses in groups.values() for gloss in glosses}
    listed |= {piece.lower() for part in _split_top(brief, ";") for piece in _split_top(part, ",")}
    senses = []
    for tag, (name, help_text) in PARTS_OF_SPEECH.items():
        if tag in ("vt", "vi"):
            continue
        # A verb's meanings come in one block: any it has as a plain verb, then with and without an object.
        lines = [("", tag)] + ([("with an object", "vt"), ("without an object", "vi")] if tag == "v" else [])
        lines = [{"label": label, "glosses": groups[key]} for label, key in lines if key in groups]
        if lines:
            senses.append({"pos": name, "help": help_text, "lines": lines})
    return {
        "word": word,
        "usage": raw.get("usage", ""),
        "brief": brief,
        "senses": senses,
        # The English the community gives it most often, leaving out what the entry already lists
        "often": [en for en in list(raw.get("ku_data") or {})[:10] if en.lower() not in listed],
        "see_also": raw.get("see_also") or [],
    }


def _notes(diagnostics) -> list[dict]:
    return [{"severity": d.severity, "code": d.code, "message": d.message, "words": d.source} for d in diagnostics]


_renderer: EnglishRenderer | None = None


def analyse(text: str) -> str:
    global _renderer
    _renderer = _renderer or EnglishRenderer()
    out = []
    for result in validate_text(text):
        readings = []
        seen = set()
        for utterance, probability in result.readings:
            reading = {
                "english": _renderer.utterance(utterance, result.text),
                "probability": round(probability, 3),
                "tree": _tree(utterance),
            }
            # Two parses can build the same structure; show it once.
            key = json.dumps([reading["english"], reading["tree"]])
            if key in seen:
                continue
            seen.add(key)
            diagnostics = result.diagnostics if not readings else diagnose(result, utterance)
            reading["notes"] = _notes(diagnostics)
            reading["valid"] = not any(d.severity == "ERROR" for d in diagnostics)
            readings.append(reading)
        words = [token.text.lower() for token in tokenize(result.text)]
        out.append({
            "text": result.text,
            "readings": readings,
            # A sentence that doesn't parse still gets its words glossed one by one, beside the notes.
            "gloss": None if readings else _renderer.gloss(result.text),
            "notes": [] if readings else _notes(result.diagnostics),
            "entry": entry(words[0]) if len(words) == 1 else None,
        })
    return json.dumps(out, ensure_ascii=False)

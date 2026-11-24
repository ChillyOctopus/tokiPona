#!/usr/bin/env python3
"""
Run with .venv/bin/python3 build_wasona_corpus.py

build_wasona_corpus.py
Extracts every Toki Pona example from the wasona course into wasona_sentences.tsv.
Needs the submodule: git submodule update --init compiled_sources/wasona

Columns:
  lesson      lesson file, e.g. basics-1/03-simple-sentences
  kind        example      <Sentence sl=... m=...> in the lesson text
              word         <Word sl=... m=...> gloss (a word or short phrase)
              vocab        a one-word practice answer (a vocabulary drill)
              exercise     practice task: l2 (toki pona) for l1 (English)
              also_accept  another accepted answer to the same exercise
              text         <Row sl=... m=...> lines of a story, song or writing
                           sample; a line that carries on the one before
                           (lexer.continues_line) is joined to it
  toki_pona   the Toki Pona text (sitelen Lasina)
  english     the English meaning ('' when the lesson gives none)
"""

import csv
import glob
import html
import os
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

from lexer import continues_line

WASONA_DIR = "compiled_sources/wasona/content/en"
OUTPUT     = "wasona_sentences.tsv"

# Attribute values can hold markup such as <u>e soweli</u>, so match
# name="value" pairs instead of stopping at the first '>'.
_TAG_RE        = re.compile(r'<(Sentence|Word|Row)((?:\s+[a-zA-Z]+="[^"]*")*)\s*/?>')
_ATTR_RE       = re.compile(r'([a-zA-Z]+)="([^"]*)"')
_BREAK_RE      = re.compile(r"<br\s*/?>", re.I)
_MARKUP_RE     = re.compile(r"<[^>]+>")
_JSX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.S)
_ONE_WORD_RE   = re.compile(r"[a-z]+")


def _clean(value) -> str:
    """Strip markup and HTML entities (a <br> becomes a space), collapse whitespace."""
    text = html.unescape(_MARKUP_RE.sub("", _BREAK_RE.sub(" ", str(value))))
    return " ".join(text.split())


def _frontmatter(source: str) -> dict:
    if not source.startswith("---"):
        return {}
    end = source.find("\n---", 3)
    if end == -1:
        return {}
    return yaml.safe_load(source[3:end]) or {}


def extract(path: str) -> list[tuple[str, str, str, str]]:
    """Return (lesson, kind, toki_pona, english) rows for one lesson file."""
    lesson = os.path.splitext(os.path.relpath(path, WASONA_DIR))[0]
    with open(path, encoding="utf-8") as fh:
        source = fh.read()

    rows = []
    for block in _frontmatter(source).get("practice") or []:
        for task in block.get("tasks") or []:
            english = _clean(task.get("l1", ""))
            answers = [task.get("l2", "")] + list(task.get("alsoAccept") or [])
            for i, answer in enumerate(map(_clean, answers)):
                if answer:
                    kind = "exercise" if i == 0 else "also_accept"
                    if _ONE_WORD_RE.fullmatch(answer):
                        kind = "vocab"   # a one-word drill, not a sentence
                    rows.append((lesson, kind, answer, english))

    kinds = {"Sentence": "example", "Word": "word", "Row": "text"}
    for tag in _TAG_RE.finditer(_JSX_COMMENT_RE.sub("", source)):
        attrs = dict(_ATTR_RE.findall(tag.group(2)))
        toki_pona, english = _clean(attrs.get("sl", "")), _clean(attrs.get("m", ""))
        if not toki_pona:
            continue
        kind = kinds[tag.group(1)]
        if kind == "text" and rows and rows[-1][1] == "text" and continues_line(rows[-1][2], toki_pona):
            _, _, previous, previous_english = rows.pop()
            toki_pona = f"{previous} {toki_pona}"
            english = f"{previous_english} {english}".strip()
        rows.append((lesson, kind, toki_pona, english))
    return rows


def main():
    files = sorted(glob.glob(os.path.join(WASONA_DIR, "**", "*.mdx"), recursive=True))
    if not files:
        sys.exit(f"No wasona lessons under {WASONA_DIR}; "
                 "run: git submodule update --init compiled_sources/wasona")

    rows = list(dict.fromkeys(row for path in files for row in extract(path)))
    with open(OUTPUT, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["lesson", "kind", "toki_pona", "english"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

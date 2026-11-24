"""
web/build.py
Assembles web/dist, the static site at toki.jistevenson.com:

    (cd web && npm ci) && .venv/bin/python web/build.py

  web/src/*            the page: HTML, CSS, the UI script and the worker
  dictionary.html      every word in master_dictionary.yaml, written into web/src/dictionary.html by web/api.py's entry()
  pyodide/             Pyodide's runtime and standard library, from web/node_modules (self-hosted, so the page loads
                       nothing from anyone else)
  fonts/               Newsreader and Hanken Grotesk, as on jistevenson.com
  app.zip              the parser, validator and translator, their data, web/api.py, and the two libraries they need
                       (Lark and PyYAML, both pure Python, taken from the Python running this script)

The worker unpacks app.zip into Pyodide's file system and imports api.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from html import escape
from pathlib import Path

import lark
import yaml

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = WEB / "dist"
MODULES = WEB / "node_modules"

# The Python the page runs, relative to the repo root; each keeps its path inside app.zip
APP_FILES = [
    "ast_builder.py", "ast_nodes.py", "lexer.py", "parser.py", "validator.py",
    "master_dictionary.yaml", "wasona_sentences.tsv",
    "translation/__init__.py", "translation/choose.py", "translation/english.py", "translation/lexicon.py",
    "translation/to_english.py", "translation/compounds.tsv", "translation/names.tsv",
]
PYODIDE_FILES = ["pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
FONTS = {
    "@fontsource-variable/newsreader/files/newsreader-latin-wght-normal.woff2": "newsreader.woff2",
    "@fontsource-variable/hanken-grotesk/files/hanken-grotesk-latin-wght-normal.woff2": "hanken-grotesk.woff2",
}


def add_package(archive: zipfile.ZipFile, package: Path) -> None:
    """A pure-Python package's .py files, under its own name (compiled extensions and caches stay behind)."""
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".lark") and "__pycache__" not in path.parts:
            archive.write(path, path.relative_to(package.parent).as_posix())


# The distribution each package comes from, whose .dist-info holds its license
DISTRIBUTIONS = {"lark": "lark", "yaml": "pyyaml"}


def license_of(package: Path) -> Path:
    """The LICENSE in the package's .dist-info folder, which ships beside it in app.zip."""
    found = sorted(package.parent.glob(f"{DISTRIBUTIONS[package.name]}-*.dist-info/licenses/LICENSE*"))
    if not found:
        raise SystemExit(f"build: no LICENSE for {package.name}")
    return found[0]


# lipu Linku's usage tiers, most used first, with what each means
USAGE = {
    "core": "nearly every speaker uses it",
    "common": "most speakers use it",
    "uncommon": "some speakers use it",
    "obscure": "few speakers use it",
}


def glosses_text(glosses: list[str]) -> str:
    """A part of speech's meanings in a line: commas between them, or semicolons when one has its own commas."""
    return ("; " if any("," in g for g in glosses) else ", ").join(glosses)


def check_linku(words: list[str]) -> None:
    """Fail the build if a piece of a Linku definition has no part of speech, a row of linku_parts.tsv matches no
    piece, or a piece of Linku's text is missing from its entry."""
    from api import LINKU_PARTS, entry, linku_pieces

    used, problems = set(), []
    for word in words:
        shown = " | ".join(g for s in entry(word)["senses"] for line in s["lines"] for g in line["glosses"])
        for tag, piece in linku_pieces(word):
            used.add((word, piece))
            if tag is None:
                problems.append(f"{word}: no part of speech for {piece!r} in web/linku_parts.tsv")
            for part in piece.split(", "):
                if part.strip() not in shown:
                    problems.append(f"{word}: {part!r} is missing from the entry")
    problems += [f"{w}: {p!r} in web/linku_parts.tsv matches no Linku piece" for w, p in LINKU_PARTS if (w, p) not in used]
    if problems:
        raise SystemExit("build: " + "\n       ".join(problems))


def dictionary_page() -> str:
    """web/src/dictionary.html with every entry filled in, alphabetically, each under an id the translator links to."""
    sys.path[:0] = [str(ROOT), str(WEB)]
    from api import entry
    from lexer import ENTRIES

    words = sorted(ENTRIES)
    check_linku(words)
    by_letter: dict[str, list[str]] = {}
    for word in words:
        by_letter.setdefault(word[0], []).append(word)

    sections = []
    for letter, group in by_letter.items():
        articles = []
        for word in group:
            e, w = entry(word), escape(word)
            parts = "".join(
                f'<div class="pos"><details class="help"><summary><span class="pos-name">{escape(s["pos"])}</span>'
                f'<span class="q" aria-hidden="true">?</span><span class="visually-hidden"> (what is this?)</span>'
                f'</summary><p>{escape(s["help"])}</p></details>'
                + "".join(
                    f'<p class="glosses">'
                    + (f'<span class="line-label">{escape(line["label"])}:</span> ' if line["label"] else "")
                    + f'{escape(glosses_text(line["glosses"]))}</p>'
                    for line in s["lines"]
                )
                + "</div>"
                for s in e["senses"]
            )
            often = (f'<p class="often"><span class="often-label">Often translated as</span> '
                     f'{escape(", ".join(e["often"]))}</p>') if e["often"] else ""
            see = ", ".join(f'<a href="#{escape(x)}">{escape(x)}</a>' for x in e["see_also"] if x in ENTRIES)
            articles.append(
                f'<article class="word" id="{w}" data-brief="{escape(e["brief"])}" data-usage="{escape(e["usage"])}">'
                f'<h3><a href="#{w}">{w}</a> <span class="usage usage-{escape(e["usage"])}">{escape(e["usage"])}'
                f'</span></h3><p class="brief">{escape(e["brief"])}</p><div class="parts">{parts}</div>{often}'
                + (f'<p class="see">See also {see}</p>' if see else "")
                + "</article>"
            )
        count = f"{len(group)} word" + ("" if len(group) == 1 else "s")
        sections.append(
            f'<section class="letter-section" id="letter-{letter}" aria-labelledby="heading-{letter}">'
            f'<header class="letter-head"><h2 id="heading-{letter}">{letter}</h2><p>{count}</p>'
            f'<a class="to-top" href="#top-letters">Back to letters</a></header>{"".join(articles)}</section>'
        )

    tiers = {tier: sum(ENTRIES[w].get("usage") == tier for w in words) for tier in USAGE}
    usage = "".join(
        f'<li><span class="usage usage-{t}">{t}</span> <span>{escape(about)} '
        f'<span class="muted">({tiers[t]} words)</span></span></li>'
        for t, about in USAGE.items()
    )
    letters = "".join(f'<li><a href="#letter-{c}" aria-label="Words starting with {c}">{c}</a></li>' for c in by_letter)
    page = (WEB / "src" / "dictionary.html").read_text(encoding="utf-8")
    return (page.replace("<!-- count -->", str(len(words)))
                .replace("<!-- usage -->", usage)
                .replace("<!-- letters -->", letters)
                .replace("<!-- entries -->", "\n".join(sections)))


def stamp(name: str, *refs: str) -> None:
    """Add ?v=<hash of the file> to each quoted reference in dist/name that it makes."""
    path = DIST / name
    text = path.read_text(encoding="utf-8")
    for ref in refs:
        digest = hashlib.sha256((DIST / ref).read_bytes()).hexdigest()[:10]
        for quote in "'\"":
            text = text.replace(f"{quote}{ref}{quote}", f"{quote}{ref}?v={digest}{quote}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not (MODULES / "pyodide").exists():
        raise SystemExit("build: run `npm ci` in web/ first")
    shutil.rmtree(DIST, ignore_errors=True)
    shutil.copytree(WEB / "src", DIST)
    (DIST / "dictionary.html").write_text(dictionary_page(), encoding="utf-8")

    (DIST / "pyodide").mkdir()
    for name in PYODIDE_FILES:
        shutil.copy2(MODULES / "pyodide" / name, DIST / "pyodide" / name)
    (DIST / "fonts").mkdir()
    for source, name in FONTS.items():
        shutil.copy2(MODULES / source, DIST / "fonts" / name)

    with zipfile.ZipFile(DIST / "app.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in APP_FILES:
            archive.write(ROOT / name, name)
        archive.write(WEB / "api.py", "api.py")
        archive.write(WEB / "linku_parts.tsv", "linku_parts.tsv")
        for module in (lark, yaml):
            package = Path(module.__file__).parent
            add_package(archive, package)
            archive.write(license_of(package), f"licenses/{package.name}.txt")
        archive.write(ROOT / "LICENSE", "licenses/tokiPona.txt")

    # Browsers keep files for hours, so each reference carries a hash of what it points at: a changed file gets a new
    # URL and is fetched at once. Leaves first, so a file's hash covers the stamps inside it.
    stamp("worker.js", "./app.zip")
    stamp("app.js", "./worker.js")
    for page in ("index.html", "dictionary.html"):
        stamp(page, "style.css", "app.js", "dictionary.js")

    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"build: wrote {DIST.relative_to(ROOT)} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

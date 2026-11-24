#!/usr/bin/env python3
"""
Run with .venv/bin/python3 build_dictionary.py

build_dictionary.py
Merges all Toki Pona lexical sources into master_dictionary.yaml.

Sources (all but wasona sit in private/, which git ignores: they're other
people's work, so the repo carries only the merged result):
  - private/compiled_sources/TokiPonaData/linku.json      (authoritative metadata, ku_data)
  - private/compiled_sources/the_good_stuff_dictionary.txt (rich POS senses + preverbs)
  - private/compiled_sources/dictionary.txt               (transitive/intransitive distinction)
  - private/compiled_sources/cheat_sheet.txt              (brief canonical glosses)
  - compiled_sources/wasona/content/en/                   (pedagogical glosses from Astro course; git submodule)

Manual corrections (EXTRA_SENSES, FIELD_OVERRIDES) are applied last, so hand
fixes survive a rebuild. Add new ones there, not in the YAML.

Output:
  master_dictionary.yaml
"""

import glob
import json
import os
import re
import sys
from collections import OrderedDict

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TP_WORDS = set()  # filled after linku load; used to detect pu_alt vs preposition

def _clean_etym(etym_list):
    return [e for e in etym_list if any(e.values())]

def _top_ku(ku_dict, n=20):
    """Return ku_data sorted by frequency desc, keeping entries >= 25."""
    return OrderedDict(
        sorted(
            ((censor(k), v) for k, v in ku_dict.items() if v >= 25),
            key=lambda x: -x[1],
        )[:n]
    )

# ---------------------------------------------------------------------------
# Parser: the_good_stuff_dictionary.txt
# ---------------------------------------------------------------------------

# POS tag pattern: n n1 n2 v v1 v2 a a1 p m m1 m2 f f1 #  #1 i c
_POS_RE = re.compile(
    r'^([nvapmfic#])(\d?)\s+'      # e.g. "n1 " or "v "
    r'(.*)'
)
_META_RE = re.compile(r'^(la|pu|see|notes|Notes)\s+(.*)')

def parse_good_stuff(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    # Split into word blocks: a block starts with a bare lowercase word line
    raw_sections = re.split(r'\n(?=[a-z][a-z/]*\n)', text)
    word_sections = [s for s in raw_sections if re.match(r'^[a-z][a-z/]*\n', s)]

    entries = {}
    for block in word_sections:
        lines = block.splitlines()
        # First line is the word (may be "ale/ali")
        raw_word = lines[0].strip()
        word = raw_word.split("/")[0].strip()

        entry = {
            "senses_gs": [],
            "see_gs": [],
            "pu_alt_gs": "",
            "notes_gs": "",
        }

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            mm = _META_RE.match(line)
            if mm:
                tag, rest = mm.group(1).lower(), mm.group(2).strip()
                if tag == "la":
                    entry["senses_gs"].append({"pos": "la", "gloss": rest})
                elif tag == "pu":
                    entry["pu_alt_gs"] = rest
                elif tag in ("see",):
                    entry["see_gs"] = [s.strip() for s in rest.split(",")]
                elif tag in ("notes", "notes"):
                    entry["notes_gs"] = rest
                continue

            pm = _POS_RE.match(line)
            if pm:
                pos_base = pm.group(1)
                gloss    = pm.group(3).strip()

                # Distinguish "p" as preposition vs pu_alt:
                # If the gloss consists only of known TP words / commas / spaces, treat as pu_alt.
                if pos_base == "p":
                    gloss_words = set(re.findall(r"[a-z]+", gloss))
                    if gloss_words and gloss_words.issubset(TP_WORDS | {"ala"}):
                        entry["pu_alt_gs"] = gloss
                        continue

                entry["senses_gs"].append({"pos": pos_base, "gloss": gloss})
                continue

        entries[word] = entry

    return entries


# ---------------------------------------------------------------------------
# Parser: dictionary.txt
# ---------------------------------------------------------------------------

_DICT_POS_MAP = {
    "n":     "n",
    "mod":   "m",
    "sep":   "f",
    "vt":    "vt",
    "vi":    "vi",
    "interj":"i",
    "prep":  "p",
    "conj":  "c",
    "kama":  "kama",
    "cont":  "la",
    "oth":   "other",
}

def _dict_sense(text):
    """A {pos, gloss} sense from 'vt pollute, dirty', or None if no POS tag leads."""
    # The longest matching POS tag at the start
    for tag in sorted(_DICT_POS_MAP, key=len, reverse=True):
        if text.startswith(tag + " ") or text == tag:
            return {"pos": _DICT_POS_MAP[tag], "gloss": text[len(tag):].strip()}
    return None


def parse_dictionary(path):
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    entries = {}
    current = None

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            continue

        # Skip the "Parts of Speech:" header block
        if stripped.startswith("Parts of Speech") or stripped.startswith("- n head"):
            current = None
            continue

        # Sense line: starts with "- " (with optional leading space)
        if stripped.startswith("- ") and current:
            sense = _dict_sense(stripped[2:].strip())
            if sense:
                entries[current]["senses_d"].append(sense)
            continue

        # Bare word header: no leading space, one word or "ali / (ale)"
        if not line.startswith(" ") and not line.startswith("-"):
            header = re.match(r"^([a-z]+)(?:\s*/\s*\([a-z]+\))?$", stripped)
            if header:
                current = header.group(1)
                if current not in entries:
                    entries[current] = {"senses_d": []}
            elif current:
                # A wrapped line: a sense that lost its "- " ("n dirt, pollution,
                # filth" under jaki) or the rest of the last gloss ("manage to,
                # start to" under kama). Read as headers, they once made a word
                # "manage" and gave n the senses of jaki and pimeja.
                sense = _dict_sense(stripped)
                if sense:
                    entries[current]["senses_d"].append(sense)
                elif entries[current]["senses_d"]:
                    entries[current]["senses_d"][-1]["gloss"] += " " + stripped

    return entries


# ---------------------------------------------------------------------------
# Parser: cheat_sheet.txt (dictionary section only)
# ---------------------------------------------------------------------------

def parse_cheat_sheet(path):
    """Extract single-line glosses from the word-list section."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # The word list starts after the grammar section, lines like "a: gloss"
    glosses = {}
    for line in text.splitlines():
        m = re.match(r"^([a-z]+):\s+(.+)", line)
        if m:
            glosses[m.group(1)] = m.group(2).strip()
    return glosses


# ---------------------------------------------------------------------------
# Parser: wasona MDX lessons
# ---------------------------------------------------------------------------

_WORD_TAG_RE = re.compile(r'<Word\b([^>]*)>')
_SL_RE       = re.compile(r'\bsl="([a-z]+)"')
_M_RE        = re.compile(r'\bm="([^"]+)"')

def parse_wasona(wasona_dir):
    """
    Extract first <Word sl="word" m="gloss" /> per word from all wasona MDX lessons.
    Files are processed in sorted order (lesson progression).
    HTML tags and usage-percentage suffixes are stripped from glosses.
    """
    glosses = {}
    pattern = os.path.join(wasona_dir, "content", "en", "**", "*.mdx")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        sys.exit(f"No wasona lessons under {wasona_dir}; "
                 "run: git submodule update --init compiled_sources/wasona")

    for fpath in files:
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                # Skip JSX comment lines
                if "{/*" in stripped or stripped.startswith("//"):
                    continue
                for tag_match in _WORD_TAG_RE.finditer(stripped):
                    attrs = tag_match.group(1)
                    sl_m = _SL_RE.search(attrs)
                    m_m  = _M_RE.search(attrs)
                    if not (sl_m and m_m):
                        continue
                    word  = sl_m.group(1)
                    gloss = m_m.group(1)
                    # Strip HTML tags
                    gloss = re.sub(r"<[^>]+>", "", gloss)
                    # Strip trailing usage percentage like " (59%)"
                    gloss = re.sub(r"\s*\(\d+%\)\s*$", "", gloss).strip()
                    if word and word not in glosses:
                        glosses[word] = gloss

    return glosses


# ---------------------------------------------------------------------------
# Sense merging
# ---------------------------------------------------------------------------

# Typos in the sources' glosses, corrected as they're read
TYPOS = {
    "brake, damage": "break, damage",
    "broken, damaged, non-funcional": "broken, damaged, non-functional",
}


def censor(text):
    """Star out the f-word, keeping its shape so the meaning still reads: "f**k up", "f**king" (and the source's "fuckk")."""
    return re.sub(r"\b([Ff])uckk?", r"\1**k", text)


def _merge_senses(senses_gs, senses_d):
    """
    Merge senses from the_good_stuff (senses_gs) and dictionary.txt (senses_d).
    Normalise 'vt'/'vi' from dict into the unified sense list.
    Add a 'sources' list to each sense so readers know the provenance.
    """
    # Build a de-duplicated sense list, preserving order.
    # Key: (pos, normalised_gloss)
    seen = {}
    result = []

    def add(pos, gloss, source):
        gloss = censor(TYPOS.get(gloss, gloss))
        key = (pos, re.sub(r"\s+", " ", gloss.lower()).strip(" .,"))
        if key not in seen:
            seen[key] = len(result)
            result.append({"pos": pos, "gloss": gloss, "sources": [source]})
        else:
            src_list = result[seen[key]]["sources"]
            if source not in src_list:
                src_list.append(source)

    # good_stuff first (richer POS taxonomy)
    for s in senses_gs:
        add(s["pos"], s["gloss"], "good_stuff")

    # dictionary.txt second
    for s in senses_d:
        pos = s["pos"]
        # Normalise vt/vi → v when good_stuff already has a generic 'v'
        if pos in ("vt", "vi"):
            generic_key = ("v", re.sub(r"\s+", " ", s["gloss"].lower()).strip(" .,"))
            if generic_key in seen:
                src_list = result[seen[generic_key]]["sources"]
                if "dictionary" not in src_list:
                    src_list.append("dictionary")
                continue
        add(pos, s["gloss"], "dictionary")

    return result


# ---------------------------------------------------------------------------
# Manual corrections (applied after merging, so a rebuild keeps them)
# ---------------------------------------------------------------------------

# Preverb senses wasona teaches that the POS sources lack.
# lexer.py builds its PREVERBS set from 'a' senses.
EXTRA_SENSES = {
    "alasa": [{"pos": "a", "gloss": "to try to", "sources": ["wasona"]}],
    "sona":  [{"pos": "a", "gloss": "to know how to", "sources": ["wasona"]}],
    # wasona lesson 28: "n is an interjection just like a, but nasally"
    "n":     [{"pos": "i", "gloss": "hmm, umm", "sources": ["wasona"]}],
}

# ali is a pronunciation variant of ale; give it ale's full meaning,
# including the number sense lexer.py uses for NUMBERS.
_ALE_ALL = ("all; abundant, countless, bountiful, every, plentiful; "
            "abundance, everything, life, universe; 100")
FIELD_OVERRIDES = {
    "ale": {"definition": _ALE_ALL},
    "ali": {
        "brief": "all, every",
        "pu_alt": "ale",
        "definition": _ALE_ALL,
        "senses": [
            {"pos": "n",  "gloss": "everything, life, universe", "sources": ["good_stuff"]},
            {"pos": "m",  "gloss": "all, each, every, complete, whole, countless", "sources": ["good_stuff"]},
            {"pos": "#",  "gloss": "all, countless amount", "sources": ["good_stuff"]},
            {"pos": "#",  "gloss": "100", "sources": ["good_stuff"]},
            {"pos": "la", "gloss": "all told, nevertheless, despite everything, either way",
             "sources": ["good_stuff"]},
        ],
    },
    # Core words wasona offers in place of uncommon ones (more/28-02);
    # validator.py's uncommon-word note suggests them.
    "apeja":   {"pu_alt": "pilin anpa"},
    "epiku":   {"pu_alt": "wawa"},
    "isipin":  {"pu_alt": "toki insa, pilin"},
    "jasima":  {"pu_alt": "ilo lukin"},
    "linluwi": {"pu_alt": "kulupu ilo"},
    "majuna":  {"pu_alt": "sin ala, pi tenpo weka"},
    "meso":    {"pu_alt": "insa"},
    "oko":     {"pu_alt": "lukin"},
    "po":      {"pu_alt": "tu tu"},
    "san":     {"pu_alt": "tu wan"},
    "usawi":   {"pu_alt": "wawa, nasa, ken"},
}


def _apply_overrides(master):
    for word, senses in EXTRA_SENSES.items():
        master[word].setdefault("senses", []).extend(senses)
    for word, fields in FIELD_OVERRIDES.items():
        master[word].update(fields)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(linku_path, gs_path, dict_path, cs_path, wasona_dir, output_path):
    # Load sources
    with open(linku_path, encoding="utf-8") as fh:
        linku = json.load(fh)

    global TP_WORDS
    TP_WORDS = set(linku.keys())

    gs_entries      = parse_good_stuff(gs_path)
    dict_entries    = parse_dictionary(dict_path)
    cs_glosses      = parse_cheat_sheet(cs_path)
    wasona_glosses  = parse_wasona(wasona_dir)

    # Collect all words: linku is authoritative, but include extras from good_stuff
    all_words = sorted(set(linku.keys()) | set(gs_entries.keys()) | set(dict_entries.keys()))

    master = OrderedDict()

    for word in all_words:
        lw  = linku.get(word, {})
        gs  = gs_entries.get(word, {"senses_gs": [], "see_gs": [], "pu_alt_gs": "", "notes_gs": ""})
        dw  = dict_entries.get(word, {"senses_d": []})

        # --- Metadata ---
        book           = lw.get("book", "none")
        usage          = lw.get("usage_category", "")
        deprecated     = lw.get("deprecated", False)
        creator        = lw.get("creator", [])
        coined_era     = lw.get("coined_era", "")
        source_lang    = lw.get("source_language", "") or ""
        etymology      = _clean_etym(lw.get("etymology", []))
        see_also       = sorted(set(lw.get("see_also", []) + gs.get("see_gs", [])))
        pu_alt         = gs.get("pu_alt_gs", "") or lw.get("representations", {}).get("pu_verbatim", "")
        pu_verbatim    = lw.get("pu_verbatim", {}).get("en", "")
        definition     = lw.get("translations", {}).get("en", {}).get("definition", "")
        commentary     = lw.get("translations", {}).get("en", {}).get("commentary", "")
        ku_data        = _top_ku(lw.get("ku_data", {}))

        # --- Senses ---
        senses = _merge_senses(gs.get("senses_gs", []), dw.get("senses_d", []))

        # --- Representations ---
        reps           = lw.get("representations", {})
        ucsur          = reps.get("ucsur", "")
        sitelen_emosi  = reps.get("sitelen_emosi", "")

        # --- Usage stats ---
        usage_stats    = lw.get("usage", {})

        # --- Brief gloss: cheat sheet > wasona > first sense ---
        brief = cs_glosses.get(word, "")
        if not brief:
            brief = wasona_glosses.get(word, "")
        if not brief and senses:
            brief = senses[0]["gloss"]

        # --- Wasona gloss ---
        wasona = wasona_glosses.get(word, "")

        # --- Notes ---
        notes = gs.get("notes_gs", "") or commentary

        # --- Assemble entry ---

        entry = OrderedDict()
        entry["word"]          = word
        entry["brief"]         = brief
        # entry["book"]          = book
        entry["usage"]         = usage
        # if deprecated:
        #     entry["deprecated"] = True
        # entry["creator"]       = creator
        # entry["coined_era"]    = coined_era
        # entry["source_language"] = source_lang
        # if etymology:
        #     entry["etymology"] = etymology
        if see_also:
            entry["see_also"]  = see_also
        if pu_alt:
            entry["pu_alt"]    = pu_alt
        # pu_verbatim (the official book's definitions) stays out: that text is Sonja Lang's, not ours to republish.
        if definition:
            entry["definition"]  = definition
        if wasona:
            entry["wasona"]    = wasona
        if senses:
            entry["senses"]    = senses
        if ku_data:
            entry["ku_data"]   = dict(ku_data)
        # entry["representations"] = OrderedDict(
        #     [k for k in [("ucsur", ucsur), ("sitelen_emosi", sitelen_emosi)] if k[1]]
        # ) or {}
        # if usage_stats:
        #     entry["usage_stats"] = usage_stats
        if notes:
            entry["notes"]     = notes

        master[word] = entry

    _apply_overrides(master)

    # Write YAML
    class _Dumper(yaml.Dumper):
        pass

    # Make OrderedDict dump in insertion order
    _Dumper.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items()),
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("# ============================================================\n")
        fh.write("# MASTER TOKI PONA DICTIONARY\n")
        fh.write("# Auto-generated by build_dictionary.py\n")
        fh.write("#\n")
        fh.write("# Sources:\n")
        fh.write("#   linku.json                    authoritative metadata, ku community translations\n")
        fh.write("#   the_good_stuff_dictionary.txt  rich POS senses, preverbs, la-meanings\n")
        fh.write("#   dictionary.txt                transitive/intransitive verb distinction\n")
        fh.write("#   cheat_sheet.txt               brief canonical glosses\n")
        fh.write("#   wasona/content/en/**/*.mdx    pedagogical glosses from Astro course\n")
        fh.write("#\n")
        fh.write("# POS TAGS used in 'senses':\n")
        fh.write("#   n    noun            m    modifier (adjective or adverb)\n")
        fh.write("#   vt   transitive verb vi   intransitive verb\n")
        fh.write("#   v    verb (unspecified transitivity)\n")
        fh.write("#   a    preverb / auxiliary verb\n")
        fh.write("#   p    preposition     f    function / particle\n")
        fh.write("#   #    numeric value   i    interjection\n")
        fh.write("#   c    conjunction     la   meaning when standing alone before la\n")
        fh.write("#   kama  kama + V compound  other  uncategorised special usage\n")
        fh.write("#\n")
        fh.write("# FIELDS:\n")
        fh.write("#   word          Toki Pona word\n")
        fh.write("#   brief         one-line gloss (cheat sheet > wasona > first sense)\n")
        # fh.write("#   book          pu | ku_suli | ku_lili | none\n")
        fh.write("#   usage         core | common | uncommon | obscure\n")
        # fh.write("#   creator       who coined the word\n")
        # fh.write("#   coined_era    pre-pu | pu-era | post-pu\n")
        # fh.write("#   source_language  language of origin\n")
        # fh.write("#   etymology     list of {word, language, definition, alt}\n")
        fh.write("#   see_also      related/merged words\n")
        fh.write("#   pu_alt        pu-recommended replacement for non-pu words\n")
        fh.write("#   definition    linku community definition\n")
        fh.write("#   wasona        pedagogical gloss from the wasona Toki Pona course\n")
        fh.write("#   senses        list of {pos, gloss, sources}\n")
        fh.write("#   ku_data       community translation frequencies 0-100 (sorted desc)\n")
        # fh.write("#   representations  {ucsur, sitelen_emosi}\n")
        # fh.write("#   usage_stats   % of speakers using word per survey period\n")
        fh.write("#   notes         usage notes, history, caveats\n")
        fh.write("# ============================================================\n\n")
        yaml.dump(
            dict(master),
            fh,
            Dumper=_Dumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
            indent=2,
        )

    print(f"Wrote {len(master)} entries to {output_path}")


if __name__ == "__main__":
    build(
        linku_path  = "private/compiled_sources/TokiPonaData/linku.json",
        gs_path     = "private/compiled_sources/the_good_stuff_dictionary.txt",
        dict_path   = "private/compiled_sources/dictionary.txt",
        cs_path     = "private/compiled_sources/cheat_sheet.txt",
        wasona_dir  = "compiled_sources/wasona",
        output_path = "master_dictionary.yaml",
    )

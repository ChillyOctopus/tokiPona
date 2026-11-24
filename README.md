# toki pona parser and translator

A parser, grammar checker, and translator for [toki pona](https://en.wikipedia.org/wiki/Toki_Pona), the minimalist
language with about 130 words. Type a sentence and it tells you whether it's grammatical, why not if it isn't, how it
parses, and what it says in English.

**Try it at [toki.jistevenson.com](https://toki.jistevenson.com).** The page runs this Python in your browser with
[Pyodide](https://pyodide.org), so nothing you type leaves your device.

```
mi moku e kili.                            → I eat the fruit.
tenpo pini la mi wile tawa ma Kanata.      → In the past, I want to go to Canada.
jan pona mi li toki e ni: sina pona.       → My good person says this: you are good.
mi moku, e moku.                           → Error: a comma before e only separates objects in a list
```

The grammar follows [wasona](https://wasona.com), a modern toki pona course, wherever sources disagree.

## How it works

| Step | File | What it does |
|---|---|---|
| Words | `lexer.py` | Splits text into sentences and words, and classes each word with the 181-word `master_dictionary.yaml` |
| Grammar | `parser.py` | A [Lark](https://github.com/lark-parser/lark) Earley grammar finds every reading of a sentence and ranks them |
| Structure | `ast_builder.py`, `ast_nodes.py` | Turns the best reading into a typed tree: subject, predicates, objects, preverbs, `pi` groups, `la` contexts |
| Checks | `validator.py` | Checks the tree against wasona's rules and explains anything off, as an error, a warning, or a note |
| English | `translation/` | Renders the tree in English, construct by construct, with each word's English chosen by its role |
| Web | `web/` | The page at toki.jistevenson.com: `web/api.py` is its entry point, run in a Web Worker |

The English is literal: toki pona marks neither tense nor number, so verbs come out in the present and nouns in the
singular unless the sentence says otherwise. `python -m translation.evaluate` scores it against wasona's own
translations (chrF 43.5 over 566 sentences). English to toki pona comes next.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                     # the tests
.venv/bin/python -m translation.evaluate       # score the English against wasona's
```

```python
from validator import validate_text
from translation.to_english import translate

for result in validate_text("mi moku, e moku."):
    print(result.is_valid, [d.message for d in result.diagnostics])
print([t.english for t in translate("soweli lili li lape lon tomo mi.")])
```

The web page:

```bash
(cd web && npm ci)
.venv/bin/python web/build.py                  # assembles web/dist
node web/test.mjs                              # runs the build in Pyodide
node web/serve.mjs                             # http://localhost:4330
```

To rebuild `wasona_sentences.tsv` or the dictionary, fetch wasona first: `git submodule update --init`.

## Credits and licenses

The code is under the [MIT License](LICENSE). The sentences, glosses, and grammar rules come from
[wasona](https://wasona.com), and the dictionary also draws on [lipu Linku](https://linku.la); the files adapted from
them are under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as
[LICENSE-DATA.md](LICENSE-DATA.md) lists. Made by [Jacob Stevenson](https://jistevenson.com).

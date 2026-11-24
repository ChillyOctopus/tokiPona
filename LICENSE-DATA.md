# Data licenses

These files are adapted from other people's work, so they carry its license
rather than the MIT License that covers the code:

| File | Adapted from | License |
|---|---|---|
| `wasona_sentences.tsv` | the example sentences, exercises, and word lists of [wasona](https://github.com/wasona/wasona), extracted by `build_wasona_corpus.py` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `master_dictionary.yaml` | [lipu Linku](https://linku.la/) (definitions, usage tiers, and `ku_data`), wasona's glosses, and three community word lists, merged by `build_dictionary.py` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `web/linku_parts.tsv` | the pieces of [lipu Linku](https://linku.la/)'s definitions that don't name a part of speech, each tagged with one by hand | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `translation/compounds.tsv`, `translation/names.tsv` | wasona's English for expressions and names | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `grammars/wasona.cfg` | wasona's grammar lessons | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

wasona's content is © its authors, under CC BY-SA 4.0. If you share these
files or something made from them, credit wasona and lipu Linku and use the
same license.

The dictionary leaves out the official definitions from *Toki Pona: The
Language of Good* by Sonja Lang, which are hers. The web build bundles Lark and
PyYAML, both under the MIT License; their license texts ship inside `app.zip`.

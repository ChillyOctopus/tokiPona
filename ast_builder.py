"""
ast_builder.py
Converts a Lark parse tree (from parser.py) into a typed Toki Pona AST.

Public API:
    build_ast(tree: Tree, offset: int = 0) -> Utterance
        Convert a single ParseResult.tree to a typed AST.

    build_ast_from_text(text: str, offset: int = 0) -> list[tuple[Utterance, float]]
        Parse text and build ASTs for all interpretations.
        Returns [(utterance, probability), ...] sorted by probability descending.
        Empty list for ungrammatical input.

    offset is where the sentence starts in a longer text; it is added to
    every Word.offset so positions point into that text.

Pipeline position: lexer → parser → **ast_builder** → validator
"""
from __future__ import annotations

from dataclasses import replace

from lark import Transformer, Token as LarkToken
from lark.tree import Tree

from ast_nodes import (
    AnyPhrase, Clause, CompoundSubject, ContextPhrase, DisjunctPhrase,
    MarkedSubject, NounPhrase, Number, ObjectPhrase, OrdinalPhrase,
    PiPhrase, PrepPhrase, Predicate, Preverb, PreverbPhrase, Preposition,
    Sentence, SentenceStarter, SimplePhrase, Subject, UnmarkedSubject,
    Utterance, VerbPhrase, VerbPhraseIntransitive, VerbPhrasePrepositional,
    VerbPhraseTransitive, VerbPhraseYN, Vocative, Word,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Structural particles stripped from source text (they add no semantic content)
_STRIP_FROM_SOURCE: frozenset[str] = frozenset({
    "LI", "E", "PI", "LA", "EN", "NANPA", "O", "ANU",
})


def _tok_to_word(tok: LarkToken, offset: int = 0) -> Word:
    """Convert a Lark Token to an AST Word node; offset shifts its position."""
    return Word(
        text=str(tok),
        token_type=tok.type,
        offset=(tok.start_pos or 0) + offset,
        source=str(tok),
    )


def _src(*parts) -> str:
    """
    Build a source string from a heterogeneous mix of arguments:
      - LarkToken         → included only if not a structural particle
      - Word              → include .text
      - AST node          → include .source
      - list              → recurse into elements
      - None              → skip
    Joined with single spaces.
    """
    pieces: list[str] = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, LarkToken):
            if p.type not in _STRIP_FROM_SOURCE and str(p).strip():
                pieces.append(str(p))
        elif isinstance(p, Word):
            pieces.append(p.text)
        elif isinstance(p, list):
            sub = _src(*p)
            if sub:
                pieces.append(sub)
        elif hasattr(p, "source") and p.source:
            pieces.append(p.source)
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class _TokiPonaASTBuilder(Transformer):
    """
    Bottom-up Lark Transformer: each method is named after a grammar rule
    and receives a list of already-transformed children.

    Tokens not handled by a named method pass through as LarkToken objects.
    Rule methods that just pass through a single child return children[0].
    """

    def __init__(self, offset: int = 0):
        super().__init__()
        self._offset = offset   # where the sentence starts in a longer text

    def _word(self, tok: LarkToken) -> Word:
        return _tok_to_word(tok, self._offset)

    # -----------------------------------------------------------------------
    # Terminal leaves
    # -----------------------------------------------------------------------

    def head(self, children: list) -> Word:
        return self._word(children[0])

    def modifier(self, children: list) -> Word:
        return self._word(children[0])

    # -----------------------------------------------------------------------
    # Number
    # -----------------------------------------------------------------------

    def number(self, children: list) -> Number:
        words = [self._word(c) for c in children if isinstance(c, LarkToken)]
        return Number(words=words, source=_src(*words))

    # -----------------------------------------------------------------------
    # Phrase hierarchy (collapses trivial wrappers)
    # -----------------------------------------------------------------------

    def simple_phrase(self, children: list) -> SimplePhrase:
        head: Word = children[0]
        mods: list[Word] = [c for c in children[1:] if isinstance(c, Word)]
        return SimplePhrase(head=head, modifiers=mods, source=_src(head, *mods))

    def nanpa_phrase(self, children: list) -> SimplePhrase | OrdinalPhrase:
        # children: [SimplePhrase]  OR  [SimplePhrase, NANPA_token, Number]
        if len(children) == 1:
            return children[0]   # collapse: no ordinal
        base: SimplePhrase = children[0]
        number_node: Number = children[2]   # children[1] is NANPA token
        return OrdinalPhrase(base=base, ordinal=number_node,
                             source=_src(base, number_node))

    def pi_phrase(self, children: list) -> AnyPhrase:
        # children: [AnyPhrase]  OR  [AnyPhrase, PI_token, AnyPhrase]
        if len(children) == 1:
            return children[0]   # collapse: no pi
        left, _pi, right = children[0], children[1], children[2]
        # Extend existing PiPhrase (left-assoc flattening) rather than nesting
        if isinstance(left, PiPhrase):
            return PiPhrase(base=left.base,
                            pi_groups=left.pi_groups + [right],
                            source=_src(left, right))
        return PiPhrase(base=left, pi_groups=[right], source=_src(left, right))

    def phrase(self, children: list) -> AnyPhrase:
        return children[0]   # pure pass-through

    def disjunct_phrase(self, children: list) -> AnyPhrase | DisjunctPhrase:
        # Filter out ANU tokens; keep only phrase nodes
        phrases = [c for c in children
                   if not isinstance(c, LarkToken)]
        if len(phrases) == 1:
            return phrases[0]   # collapse: single option
        return DisjunctPhrase(options=phrases, source=_src(*phrases))

    # -----------------------------------------------------------------------
    # Preposition
    # -----------------------------------------------------------------------

    def preposition(self, children: list) -> Preposition:
        prep_tok: LarkToken = children[0]
        negated  = any(isinstance(c, LarkToken) and c.type == "ALA"
                       for c in children)
        emphasis = any(isinstance(c, LarkToken) and c.type == "A"
                       for c in children)
        return Preposition(
            word=self._word(prep_tok),
            negated=negated,
            emphasis=emphasis,
            source=_src(*[self._word(c) for c in children
                          if isinstance(c, LarkToken)
                          and c.type not in _STRIP_FROM_SOURCE]),
        )

    def preposition_yn(self, children: list) -> Preposition:
        # children: [PREP, ALA, PREP] — "lon ala lon" asks a yes/no question
        word = self._word(children[0])
        return Preposition(word=word, negated=False, emphasis=False,
                           source=f"{word.text} ala {children[2]}", question=True)

    def prep_phrase(self, children: list) -> PrepPhrase:
        prep: Preposition = children[0]
        obj:  NounPhrase  = children[1]
        return PrepPhrase(prep=prep, obj=obj, source=_src(prep, obj))

    # -----------------------------------------------------------------------
    # Object phrase
    # -----------------------------------------------------------------------

    def object_phrase(self, children: list) -> ObjectPhrase:
        # children: [E_token, NounPhrase, PrepPhrase*]
        np: NounPhrase = children[1]
        preps: list[PrepPhrase] = [c for c in children[2:]
                                    if isinstance(c, PrepPhrase)]
        return ObjectPhrase(np=np, preps=preps, source=_src(np, *preps))

    # -----------------------------------------------------------------------
    # Preverb
    # -----------------------------------------------------------------------

    def preverb(self, children: list) -> Preverb:
        prev_tok: LarkToken = children[0]
        negated  = any(isinstance(c, LarkToken) and c.type == "ALA"
                       for c in children)
        emphasis = any(isinstance(c, LarkToken) and c.type == "A"
                       for c in children)
        src = str(prev_tok) + (" ala" if negated else "")
        return Preverb(word=self._word(prev_tok),
                       negated=negated,
                       emphasis=emphasis,
                       source=src)

    def preverb_yn(self, children: list) -> Preverb:
        # children: [PREV, ALA, PREV] — "wile ala wile" asks a yes/no question
        word = self._word(children[0])
        return Preverb(word=word, negated=False, emphasis=False,
                       source=f"{word.text} ala {children[2]}", question=True)

    # -----------------------------------------------------------------------
    # Verb phrase variants
    # -----------------------------------------------------------------------

    def verb_phrase_transitive(self, children: list) -> VerbPhraseTransitive:
        verb: AnyPhrase           = children[0]
        objects: list[ObjectPhrase] = [c for c in children[1:]
                                        if isinstance(c, ObjectPhrase)]
        preps: list[PrepPhrase]   = [c for c in children[1:]
                                      if isinstance(c, PrepPhrase)]
        return VerbPhraseTransitive(verb=verb, objects=objects, preps=preps,
                                    source=_src(verb, *objects, *preps))

    def verb_phrase_intransitive(self, children: list) -> VerbPhraseIntransitive:
        verb: AnyPhrase         = children[0]
        preps: list[PrepPhrase] = [c for c in children[1:]
                                    if isinstance(c, PrepPhrase)]
        return VerbPhraseIntransitive(verb=verb, preps=preps,
                                      source=_src(verb, *preps))

    def verb_phrase_prepositional(self, children: list) -> VerbPhrasePrepositional:
        preps: list[PrepPhrase] = [c for c in children
                                    if isinstance(c, PrepPhrase)]
        return VerbPhrasePrepositional(preps=preps, source=_src(*preps))

    def verb_phrase_yn(self, children: list) -> VerbPhraseYN:
        # children: [AnyPhrase, ALA_token, AnyPhrase, ObjectPhrase*, PrepPhrase*]
        verb: AnyPhrase = children[0]
        # children[1] = ALA token; children[2] = repeated verb phrase (stored for validation)
        repeated = children[2] if len(children) > 2 and not isinstance(children[2], ObjectPhrase) else None
        objects: list[ObjectPhrase] = [c for c in children[3:]
                                        if isinstance(c, ObjectPhrase)]
        preps: list[PrepPhrase]     = [c for c in children[3:]
                                        if isinstance(c, PrepPhrase)]
        return VerbPhraseYN(verb=verb, repeated_verb=repeated, objects=objects,
                            source=_src(verb, *objects, *preps), preps=preps)

    def verb_phrase(self, children: list) -> VerbPhrase:
        return children[0]   # pass-through to concrete variant

    # -----------------------------------------------------------------------
    # Preverb phrase (right-recursive flattening)
    # -----------------------------------------------------------------------

    def preverb_phrase(self, children: list) -> Predicate:
        if len(children) == 1:
            return children[0]   # bare VerbPhrase, no preverbs
        pv: Preverb   = children[0]
        inner: Predicate = children[1]
        if isinstance(inner, PreverbPhrase):
            # Flatten: prepend pv onto existing preverb list
            return PreverbPhrase(preverbs=[pv] + inner.preverbs,
                                 vp=inner.vp,
                                 source=_src(pv, inner))
        # inner is a bare VerbPhrase
        return PreverbPhrase(preverbs=[pv], vp=inner, source=_src(pv, inner))

    # -----------------------------------------------------------------------
    # Predicate lists (serial predicates)
    # -----------------------------------------------------------------------

    def _flatten_predicates(self, children: list) -> list[Predicate]:
        """Return only Predicate nodes, dropping LI / O / ANU tokens."""
        return [c for c in children
                if not isinstance(c, LarkToken)]

    def predicate_list_li(self, children: list) -> list[Predicate]:
        return self._flatten_predicates(children)

    def predicate_list_none(self, children: list) -> list[Predicate]:
        return self._flatten_predicates(children)

    def predicate_list_o(self, children: list) -> list[Predicate]:
        return self._flatten_predicates(children)

    # -----------------------------------------------------------------------
    # Subjects
    # -----------------------------------------------------------------------

    def unmarked_subject(self, children: list) -> UnmarkedSubject:
        w = self._word(children[0])
        return UnmarkedSubject(word=w, source=w.text)

    def marked_subject(self, children: list) -> MarkedSubject:
        np: NounPhrase = children[0]
        return MarkedSubject(phrase=np, source=np.source)

    def compound_subject(self, children: list) -> CompoundSubject:
        parts  = [c for c in children if not isinstance(c, LarkToken)]
        tokens = [c for c in children if isinstance(c, LarkToken)]
        # Flatten nested CompoundSubject from right-recursion
        flat: list[NounPhrase] = []
        for p in parts:
            if isinstance(p, CompoundSubject):
                flat.extend(p.parts)
            else:
                flat.append(p)
        # Determine conjunction from first joining token
        conj: str = "en"
        for t in tokens:
            if t.type == "EN":
                conj = "en"; break
            if t.type == "ANU":
                conj = "anu"; break
        return CompoundSubject(parts=flat, conjunction=conj,   # type: ignore[arg-type]
                               source=_src(*flat))

    def general_subject(self, children: list) -> Subject:
        return children[0]   # UnmarkedSubject or MarkedSubject

    # -----------------------------------------------------------------------
    # Clause
    # -----------------------------------------------------------------------

    def _build_clause(self, children: list) -> Clause:
        """
        Shared logic for clause_any and clause_strict.
        children is a mix of: Subject | list[Predicate] | LarkToken (LI / O / ANU)
        """
        subject:    Subject | None    = None
        predicates: list[Predicate]   = []
        has_li = False
        has_o  = False

        for child in children:
            if isinstance(child, (UnmarkedSubject, MarkedSubject, CompoundSubject)):
                subject = child
            elif isinstance(child, list):
                predicates = child   # predicate_list result
            elif isinstance(child, LarkToken):
                if child.type == "LI":
                    has_li = True
                elif child.type == "O":
                    has_o  = True

        predicate_type = "li" if has_li else "o" if has_o else "none"
        src = _src(subject, *predicates) if subject else _src(*predicates)
        return Clause(subject=subject,
                      predicate_type=predicate_type,   # type: ignore[arg-type]
                      predicates=predicates,
                      source=src)

    def clause_any(self, children: list) -> Clause:
        return self._build_clause(children)

    def clause_strict(self, children: list) -> Clause:
        return self._build_clause(children)

    # -----------------------------------------------------------------------
    # Context, vocative, sentence starters
    # -----------------------------------------------------------------------

    def context_clause(self, children: list) -> ContextPhrase:
        # children[-1] is the LA token; children[0] is Subject or Clause
        content = children[0]
        return ContextPhrase(content=content, source=content.source)

    def vocative(self, children: list) -> Vocative:
        # children: [Subject, O_token]  — strip O and any trailing A tokens
        addressee = children[0]
        return Vocative(addressee=addressee, source=addressee.source)

    def sentence_starter(self, children: list) -> SentenceStarter:
        words = [self._word(c) for c in children if isinstance(c, LarkToken)]
        return SentenceStarter(words=words, source=_src(*words))

    # -----------------------------------------------------------------------
    # Sentence and utterance (top level)
    # -----------------------------------------------------------------------

    def sentence(self, children: list) -> Sentence:
        contexts:    list[ContextPhrase] = []
        clause:      Clause | None       = None
        elaboration: Sentence | None     = None
        anu_seme = False

        for child in children:
            if isinstance(child, ContextPhrase):
                contexts.append(child)
            elif isinstance(child, Clause):
                clause = child
            elif isinstance(child, Sentence):
                elaboration = child   # the sentence after a colon
            elif isinstance(child, LarkToken):
                # ANU or SEME signals "anu seme" question
                if child.type in ("ANU", "SEME"):
                    anu_seme = True

        assert clause is not None, "sentence() received no Clause child"
        return Sentence(starters=[],    # filled in by start()
                        contexts=contexts,
                        clause=clause,
                        anu_seme=anu_seme,
                        source=_src(*contexts, clause, elaboration),
                        elaboration=elaboration)

    def elaboration(self, children: list) -> Sentence:
        # children: [SentenceStarter*, Sentence] — what follows a colon
        starters = [c for c in children if isinstance(c, SentenceStarter)]
        body: Sentence = children[-1]
        return replace(body, starters=starters, source=_src(*starters, body))

    def start(self, children: list) -> Utterance:
        starters  = [c for c in children if isinstance(c, SentenceStarter)]
        vocatives = [c for c in children if isinstance(c, Vocative)]
        sentence  = next((c for c in children if isinstance(c, Sentence)), None)

        if sentence is None and not vocatives:
            # Bare interjection: sentence_starter+ or bare O A*
            return Utterance(starters=starters, body=None, source=_src(*starters))

        if sentence is None:
            # A call on its own — starters stay at Utterance level
            body = vocatives[0]
            return Utterance(starters=starters, body=body, source=_src(*starters, body))

        # Move starters, and a call before or after the sentence, into the Sentence node
        vocative = vocatives[0] if vocatives else None
        body = replace(sentence, starters=starters, vocative=vocative,
                       source=_src(*starters, vocative, sentence))
        return Utterance(starters=[], body=body, source=body.source)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ast(tree: Tree, offset: int = 0) -> Utterance:
    """
    Convert a Lark parse tree (ParseResult.tree from parser.parse()) to a
    typed Utterance AST.

    Args:
        tree:   A Lark Tree with root data == 'start'.
        offset: Where the sentence starts in a longer text; added to every Word.offset.

    Returns:
        Utterance — the top-level AST node.

    Raises:
        ValueError: if the tree cannot be transformed.
    """
    try:
        result = _TokiPonaASTBuilder(offset).transform(tree)
    except Exception as exc:
        raise ValueError(f"AST build failed: {exc}") from exc
    if not isinstance(result, Utterance):
        raise ValueError(f"Expected Utterance root, got {type(result).__name__}")
    return result


def build_ast_from_text(text: str, offset: int = 0) -> list[tuple[Utterance, float]]:
    """
    Parse a Toki Pona string and build ASTs for all grammatical interpretations.

    Returns:
        List of (Utterance, probability) pairs sorted by probability descending.
        Empty list for empty or ungrammatical input.
    """
    from parser import parse as _tp_parse
    results = _tp_parse(text)
    out: list[tuple[Utterance, float]] = []
    for pr in results:
        try:
            ast = build_ast(pr.tree, offset)
            out.append((ast, pr.probability))
        except (ValueError, AssertionError):
            continue
    return out


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
        "taso ona li pona",
        "mi wile ala pali",
        "jan nanpa wan li pona",
    ]

    for sentence in _DEMO:
        results = build_ast_from_text(sentence)
        if not results:
            print(f"  {sentence!r}  →  NO PARSE")
            continue
        utt, prob = results[0]
        body_type = type(utt.body).__name__
        if isinstance(utt.body, Sentence):
            clause = utt.body.clause
            subj = type(clause.subject).__name__ if clause.subject else "None"
            preds = [type(p).__name__ for p in clause.predicates]
            starters = [w.text for s in utt.body.starters for w in s.words]
            print(f"  {sentence!r}")
            print(f"    p={prob:.3f}  body={body_type}"
                  + (f"  starters={starters}" if starters else ""))
            print(f"    clause: predicate_type={clause.predicate_type!r}"
                  f"  subject={subj}"
                  f"  predicates={preds}")
        else:
            print(f"  {sentence!r}  →  {body_type}  p={prob:.3f}")
        print()

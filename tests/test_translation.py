"""
tests/test_translation.py
Toki Pona -> English (translation/). The sentence tests come in two kinds:
wasona's own sentences whose English the translator already matches word for
word (the lesson is named beside each), and constructs wasona translates more
freely, where the expectation is the translator's literal rendering.
"""
from __future__ import annotations

import pytest

from translation import english
from translation.choose import FirstChoice
from translation.evaluate import _normalise, chrf, evaluate, mean
from translation.lexicon import load
from translation.to_english import translate


def en(text: str) -> str:
    return " ".join(t.english for t in translate(text))


# ---------------------------------------------------------------------------
# English word forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word,form", [
    ("animal", "animals"), ("person", "people"), ("box", "boxes"), ("city", "cities"), ("fish", "fish"),
])
def test_plural(word, form):
    assert english.plural(word) == form

@pytest.mark.parametrize("verb,agreement,form", [
    ("eat", "3s", "eats"), ("eat", "pl", "eat"), ("go", "3s", "goes"), ("try", "3s", "tries"),
    ("be", "1s", "am"), ("be", "3s", "is"), ("be", "2", "are"), ("look at", "3s", "looks at"),
    ("can", "3s", "can"),
])
def test_present(verb, agreement, form):
    assert english.present(verb, agreement) == form

@pytest.mark.parametrize("verb,form", [
    ("eat", "eating"), ("make", "making"), ("run", "running"), ("see", "seeing"), ("die", "dying"),
])
def test_ing(verb, form):
    assert english.ing(verb) == form

def test_adverbs_and_articles():
    assert english.adverb("good") == "well"
    assert english.adverb("strange") == "strangely"
    assert english.indefinite("animal") == "an animal"
    assert english.indefinite("bird") == "a bird"
    assert english.is_mass("much money") and not english.is_mass("bird")


# ---------------------------------------------------------------------------
# Lexicon and chooser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word,role", [
    ("pona", "adjective"),   # "waso li pona." = The bird is good.
    ("moku", "verb"),        # "jan li moku." = The person eats.
    ("jan", "noun"),         # "ona li jan." = They are a person.
    ("lukin", "verb"),
    ("weka", "adjective"),   # "sona li weka" = knowledge is gone, not "removes"
])
def test_predicate_role(word, role):
    assert load().get(word).predicate == role

def test_wasona_glosses_come_first():
    lexicon = load()
    assert lexicon.get("moku").options("verb")[0] == "eat"   # drill: "to eat"
    assert lexicon.get("pona").options("adjective")[0] == "good"
    assert lexicon.get("waso").options("noun")[0] == "bird"

def test_tables():
    lexicon = load()
    assert lexicon.compounds["noun"][("telo", "nasa")] == "alcohol"   # basics-2/13-elements
    assert lexicon.compounds["context"][("ken",)] == "perhaps"         # basics-3/20-time
    assert lexicon.names["Tosi"] == ("Germany", "German")
    assert lexicon.utterances["toki!"] == "Hello! Let's talk!"

def test_first_choice_keeps_the_lexicon_order():
    ranked = FirstChoice().rank("pona", "adjective", ["good", "pleasant", "nice"], "")
    assert [option for option, _ in ranked] == ["good", "pleasant", "nice"]
    assert abs(sum(p for _, p in ranked) - 1) < 1e-9


# ---------------------------------------------------------------------------
# wasona's sentences, matched word for word
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toki_pona,expected", [
    ("waso li pona.", "The bird is good."),                                   # more/06-01-li-and-is
    ("soweli li sona e pipi.", "The animal knows the bug."),                  # basics-1/03-simple-sentences
    ("mi lukin e ona.", "I see them."),                                       # basics-1/06-pronouns
    ("ona li moku pona.", "They eat well."),                                  # basics-1/06-pronouns
    ("ni li soweli.", "This is an animal."),                                  # basics-1/06-pronouns
    ("waso sina li lukin e mi.", "Your bird sees me."),                       # basics-1/06-pronouns
    ("jan Nina li toki Kanse.", "Nina speaks French."),                       # basics-1/07-names
    ("mi lukin e sitelen Kililisa.", "I see Cyrillic writing."),              # basics-1/07-names
    ("jan Mili li tan ma Kanse.", "Millie is from France."),                  # basics-1/08-prepositions
    ("ma tomo Towano li lon ma Kanata.", "Toronto is in Canada."),            # basics-1/08-prepositions
    ("waso lili li wile lape.", "The small bird wants to sleep."),            # basics-1/09-activities
    ("mi alasa tawa ma Sonko.", "I try to go to China."),                     # basics-1/09-activities
    ("jan Ema o alasa lape!", "Emma, try to sleep!"),                         # basics-2/12-requests
    ("o toki tawa jan Nina!", "Talk to Nina!"),                               # basics-2/12-requests
    ("jan Ana li tawa ala ma Tosi.", "Anna does not go to Germany."),         # basics-2/14-negation
    ("sina ken ala lape.", "You cannot sleep."),                              # basics-2/14-negation
    ("ma Masu li lon ala ma Elopa.", "Egypt is not in Europe."),              # basics-2/14-negation
    ("mi kepeken lipu Wikipesija.", "I use Wikipedia."),                      # basics-2/15-with-like
    ("ona li sama sina.", "They are like you."),                              # basics-2/15-with-like
    ("ni li leko ala leko?", "Is this a brick?"),                             # basics-2/17-questions
    ("mi lon ala lon ma Italija?", "Am I in Italy?"),                         # basics-2/17-questions
    ("sina toki Tosi anu seme?", "You speak German, right?"),                 # basics-2/17-questions
    ("waso li lon seme?", "Where is the bird?"),                              # basics-2/17-questions
    ("jan seme li tan ma Palata?", "Who is from India?"),                     # basics-2/17-questions
    ("sina wile moku e seme?", "What do you want to eat?"),                   # basics-2/17-questions
    ("luka tu wan", "Eight."),                                                # basics-3/25-numbers
    ("mi wile lape. taso mi ken ala.", "I want to sleep. But I cannot."),     # basics-3/28-moo
    ("jan li musi Manka.", "The person plays Minecraft."),                    # more/08-02-named-verbs
    ("mi pini pali.", "I finish working."),                                   # more/09-01-more-preverbs
    ("mute ale mute luka", "2025."),                                          # more/25-01-large-numbers
    ("ale ale ale ale ale ale ale ale ale ale ale ale ale ale ale", "1500."),  # more/25-01-large-numbers
])
def test_wasona_sentences(toki_pona, expected):
    assert en(toki_pona) == expected


# ---------------------------------------------------------------------------
# Constructs wasona words more freely
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toki_pona,expected", [
    ("sina moku ala moku?", "Do you eat?"),                       # V ala V
    ("sina moku e seme?", "What do you eat?"),                    # seme as the object
    ("seme li moku e pipi?", "What eats the bug?"),               # seme as the subject
    ("o moku ala!", "Do not eat!"),                               # a negative request
    ("ken la ona o lape.", "Perhaps, they should sleep."),        # a fixed context; o after ona
    ("sina wile la, sina ken.", "When you want, you can."),       # a clause before la
    ("jan Nowa o, soweli sina li wile lape!", "Noah, your animal wants to sleep!"),   # a call
    ("soweli pi suwi mute", "A very cute animal."),               # pi before an adjective group
    ("sitelen pi tomo mute", "A drawing of many buildings."),     # pi before a noun group
    ("waso suwi lili mi", "My small cute bird."),                 # modifiers in reverse
    ("jan nanpa wan li kama.", "The first person comes."),        # an ordinal
    ("jan tu li lape.", "Two people sleep."),                     # a number makes a plural
    ("jan li lape li moku.", "The person sleeps and eats."),      # two predicates
    ("pan li jo e soweli e kili e namako.", "The bread holds the animal, the fruit and the spice."),
    ("kasi li pona tawa mi.", "I like the plant."),               # "pona tawa" (lesson 15: "I like plants.")
    ("ni li pona mi.", "This is my goodness."),                # a pronoun makes it possessive, never "mily"
    ("sina jan pona mi.", "You are my friend."),                # jan pona = friend
    ("toki pona li pona.", "toki pona is good."),               # the language's name, lower case
    ("mi lukin e mi.", "I see myself."),                          # a reflexive object
    ("o kama sona e toki!", "Learn the speech!"),                 # kama sona = learn (lesson 9)
    ("kama pona!", "Welcome!"),                                   # basics-2/18-feelings
    ("toki!", "Hello! Let's talk!"),                              # basics-2/18-feelings
    ("ijo li toki e ni: suno li lon.", "Something says this: the sun is real."),   # a colon
    ("jan li toki e toki Inli.", "The person talks about English."),   # more/08-02: "speaks about English"
    ("lon tenpo kama", "In the future."),                         # a compound with its own "the"
])
def test_constructs(toki_pona, expected):
    assert en(toki_pona) == expected

def test_unparseable_text_is_glossed_word_by_word():
    [t] = translate("e moku")
    assert not t.parsed and t.english


# ---------------------------------------------------------------------------
# Evaluation against wasona
# ---------------------------------------------------------------------------

def test_chrf():
    assert chrf("The bird is good.", "the bird is good") == pytest.approx(100.0)
    assert chrf("The bird eats bugs.", "The bird eats this bug.") > chrf("A fish swims.", "The bird eats this bug.")
    assert _normalise("Where is the bird? (lit. 'The bird is at what?')") == "whereisthebird"

# Recorded on 2026-11-01. Raise them when the translator improves; a drop is a regression.
_CHRF_BASELINE = 43.5
_EXACT_BASELINE = 66

@pytest.mark.slow
def test_wasona_translation_score():
    scored = evaluate()
    exact = sum(_normalise(s.english) == _normalise(s.reference) for s in scored)
    assert round(mean(scored), 1) >= _CHRF_BASELINE, f"chrF fell to {mean(scored):.1f}"
    assert exact >= _EXACT_BASELINE, f"only {exact} translations match wasona's English"

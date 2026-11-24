"""
translation — Toki Pona to English (English to Toki Pona comes next).

    from translation.to_english import translate
    translate("waso li pona.")[0].english   # -> "The bird is good."

lexicon.py holds each word's English options, choose.py picks among them,
to_english.py renders a parse as English, english.py inflects English words,
and evaluate.py scores the output against wasona's own translations.
"""

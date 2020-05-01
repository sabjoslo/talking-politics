"""Script to clean the eng_news_2016_1M corpus, available from
http://wortschatz.uni-leipzig.de/en/download/#corporaDownload.

(c) 2018 Abteilung Automatische Sprachverarbeitung, Universitat Leipzig.
"""
import pandas as pd
import pickle
from pyspan.config import *
from pyspan.utils import *
input_dir = paths["input_dir"]

word_file = "{}leipzig_corpus/eng_news_2016_1M/eng_news_2016_1M-words.txt".format(input_dir)

with open(word_file, "r") as rfh:
    words = [ l.split("\t") for l in rfh.read().split("\n") ]
words = [ words_[-2:] for words_ in words if any([ w.strip() for w in words_ ]) ]
df = pd.DataFrame(words, columns = [ "word", "frequency" ]).set_index("word")

df["frequency"] = map(int, df["frequency"])
# Compute probabilities
denom = sum(df["frequency"])
df["probability"] = df["frequency"]/float(denom)
assert approx_eq(sum(df["probability"]), 1)

# Save
df_file = "{}leipzig_corpus/en_word_freqs".format(input_dir)
with open(df_file, "wb") as wfh:
    pickle.dump(df, wfh)

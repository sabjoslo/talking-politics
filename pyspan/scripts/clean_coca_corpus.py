"""Script to clean the COCA corpus of ngrams, available from
https://www.ngrams.info/.
"""
import pandas as pd
import pickle
import sys
from pyspan.config import *
from pyspan.utils import *
input_dir = paths["input_dir"]

n = sys.argv[1]
word_file = "{}coca/non_case_sensitive/w{}_.txt".format(input_dir, n)

with open(word_file, "r") as rfh:
    words = [ l.split("\t") for l in rfh.read().split("\n") ]
words = [ map(lambda w: w.strip(), words_) for words_ in words if
          any([ w.strip() for w in words_ ]) ]
df = pd.DataFrame(words, columns = [ "frequency", "word1", "word2" ])
df["word"] = [ "{} {}".format(w1, w2) for w1, w2 in
                df[["word1", "word2"]].values ]
df = df.set_index("word")

df["frequency"] = map(int, df["frequency"])
# Compute probabilities
denom = sum(df["frequency"])
df["probability"] = df["frequency"]/float(denom)
assert approx_eq(sum(df["probability"]), 1)

# Save
df_file = "{}coca/{}_grams".format(input_dir, n)
with open(df_file, "wb") as wfh:
    pickle.dump(df, wfh)

#!/usr/bin/python2.7

"""Generates cosine similarities of 10,000 randomly selected pairs of words so
we can get an idea of the distribution of similarities.
"""

from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import pickle
import sys
from pyspan.config import *
INPUT_DIR = paths["input_dir"]
METRICS_DIR = paths["metrics_dir"]
percentile = settings["percentile"]

metric=sys.argv[1]
mode="unigrams"

with open("{}{}-{}".format(METRICS_DIR, metric, mode)) as rfh:
    df=pickle.load(rfh)

words = df.index
words1 = np.random.choice(words, size = 10000)
words2 = np.random.choice(words, size = 10000)

word_vectors=KeyedVectors.load(INPUT_DIR + "crec_w2v")

if __name__=="__main__":
    pairs=[]
    for w1, w2 in zip(words1, words2):
        cossim=word_vectors.similarity(w1, w2)
        pairs.append((w1, w2, cossim))

    pairs=list(reversed(sorted(pairs, key=lambda x:x[2])))

    df=pd.DataFrame(pairs,
                    columns=[ "word1", "word2", "cos_sim" ])
    df.to_pickle("{}cos_sim-{}".format(METRICS_DIR, mode))

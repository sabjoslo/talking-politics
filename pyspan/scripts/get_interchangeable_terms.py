#!/usr/bin/python2.7

"""Used for stimulus generation for study 2 in "Two kinds of discussion and types of conversation".
"""

from gensim.models import KeyedVectors
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

dsorted=df.drop(columns=["rmetric"]).sort_values(["dmetric"], ascending=False)
rsorted=df.drop(columns=["dmetric"]).sort_values(["rmetric"], ascending=False)

word_vectors=KeyedVectors.load(INPUT_DIR + "crec_w2v")

lastix=int(round(len(dsorted)*percentile))

if __name__=="__main__":
    pairs=[]
    for dterm in dsorted.index[:lastix]:
        for rterm in rsorted.index[:lastix]:
            if dterm==rterm:
                continue
            cossim=word_vectors.similarity(dterm, rterm)
            pairs.append((dterm, rterm, cossim, dsorted.loc[dterm]["dmetric"],
                          rsorted.loc[rterm]["rmetric"]))

    pairs=list(reversed(sorted(pairs, key=lambda x:x[2])))

    df=pd.DataFrame(pairs,
                    columns=[ "dterm", "rterm", "cos_sim", "dmetric", "rmetric" ])
    df.to_pickle("{}{}_{}-{}".format(METRICS_DIR, "cos_sim", metric, mode))

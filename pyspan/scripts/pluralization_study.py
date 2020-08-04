#!/usr/bin/python2.7
"""Sample sets of pairs of positive, very negative and neutrally-valenced words
evenly across the Democrat distribution, where one word in each pair is a
pluralization of the other.
"""

from itertools import chain
import numpy as np
import os
import pandas as pd
import pickle
import signal
import sys
import spacy
nlp = spacy.load("en")
from pyspan.config import *
INPUT_DIR = paths["input_dir"]
METRICS_DIR = paths["metrics_dir"]
PKG_DIR = paths["package_dir"]
from pyspan.valence import get_valence

# Load relative frequency data
df = pickle.load(open(METRICS_DIR + "partial_kls-unigrams", "rb"))

df_pkl_fn = "Plurals-partial_kls_lemmas"
df_words_fn = "Plurals-stimuli"
csv_words_fn = "Plurals-survey_terms.csv"
exclude_fn = "Plurals-exclude"
study_dir = PKG_DIR + "../LoP_Plurals/"
INPUT_DIR = study_dir + "survey_materials/"

magnitude_words = [ ("second", "minute"), ("hour", "day"), ("week", "month"),
                    ("year", "decade"), ("century", "millenium"),
                    ("inch", "foot"), ("yard", "mile"),
                    ("centimeter", "kilometer"), ("ounce", "pound"),
                    ("gram", "kilogram"), ("teaspoon", "tablespoon"),
                    ("cup", "liter"), ("quart", "gallon"),
                    ("watt", "horsepower"), ("byte", "kilobyte"),
                    ("megabyte", "gigabyte"), ("hertz", "gigahertz"),
                    ("cent", "dollar"), ("one", "ten"), ("hundred", "thousand"),
                    ("few", "some"), ("several", "many"), ("small", "big"),
                    ("short", "long"), ("narrow", "wide") ]

# Find pairs with the same lemma
def get_lemma(token):
    token = unicode(token)
    token = nlp(token)[0]
    # If it's not a noun, we're not interested
    if token.pos_ != "NOUN":
        return np.nan
    return token.lemma_

def format_df():
    dem_words = df.drop(columns=["rmetric", "dmetric_std",
                                 "rmetric_std"]).sort_values(["dmetric"],
                        ascending=False)
    dem_words.reset_index(inplace = True)

    # Get lemmas, and drop everything that's not a noun
    print "Getting lemmas"
    dem_words["lemma"] = map(get_lemma, dem_words["term"])
    dem_words.dropna(subset = [ "lemma" ], inplace = True)

    # Get valence scores
    print "Getting valence scores"
    dem_words["valence"] = map(lambda w: get_valence(w)[0], dem_words["lemma"])
    dem_words.dropna(subset = [ "valence" ], inplace = True)

    with open(INPUT_DIR + df_pkl_fn, "wb") as f:
        pickle.dump(dem_words, f)

def get_pluralables():
    def _sigint_handler(signal_, frame):
        print "\nKeyboard interrupt. Cleaning up and saving to file."
        save_objs(pluralables, exclude)
        sys.exit(0)

    dem_words = pickle.load(open(INPUT_DIR + df_pkl_fn, "rb"))
    n = len(df)

    # Divide the Democrat distribution into 25 equal bins
    bins = np.arange(.04, 1.04, .04)
    assert len(bins) == 25

    signal.signal(signal.SIGINT, _sigint_handler)
    pluralables = { "pos_sm": {}, "pos_lg": {},
                    "neg_sm": {}, "neg_lg": {},
                    "neu_sm": {}, "neu_lg": {} }
    exclude = []

    # Words that have already been classified
    if os.path.exists(INPUT_DIR + df_words_fn):
        pl_df = pickle.load(open(INPUT_DIR + df_words_fn, "rb"))
        pluralables = pl_df.to_dict()

    # Words showing up in the list of plurals that either
    ## Aren't words ("vawa")
    ## If coded as neutral, could plausibly be valenced ("polluters")
    ## Don't have a natural pluralization ("hospitality")
    ## Are already included in singular/plural form ("abuse")
    ## Change meaning when you add an "s" ("good")
    ## Are verbs or nouns ("hope")
    ## Repeat in a slightly different form ("rape"/"rapist")
    ## Have multiple meanings ("field")
    if os.path.exists(INPUT_DIR + exclude_fn):
        exclude = open(INPUT_DIR + exclude_fn, "r").read().split()

    for bin_ in bins:
        if bin_ in pluralables["pos_sm"].keys():
            continue
        start, end = int(round(n*(bin_ - .04))), int(round(n*bin_))
        dem_words_ = dem_words.loc[[ ix for ix in dem_words.index if start <=
                                     ix < end]]

        # Get positive words
        pos_candidates = [ w for w in sorted(dem_words_["term"],key = lambda t:
                           abs(dem_words_.loc[dem_words_["term"] == t]["valence"].values[0]
                           - np.max(dem_words_["valence"]))) if w not in exclude
                         ]
        for word in pos_candidates:
            pair = raw_input(word + " (POS): ")
            if len(pair) == 0:
                exclude.append(word)
                continue
            sm, lg = pair.split()
            if ( sm in chain(*[ d.values() for d in pluralables.values() ])
                 or lg in chain(*[ d.values() for d in pluralables.values() ]) ):
                exclude.append(word)
                continue
            if sm not in df.index or lg not in df.index:
                exclude.append(word)
                continue
            pluralables["pos_sm"][bin_] = sm
            pluralables["pos_lg"][bin_] = lg
            break

        # Get negative words
        neg_candidates = [ w for w in sorted(dem_words_["term"],key = lambda t:
                           abs(dem_words_.loc[dem_words_["term"] == t]["valence"].values[0]
                           - np.min(dem_words_["valence"]))) if w not in exclude
                         ]
        for word in neg_candidates:
            pair = raw_input(word + " (NEG): ")
            if len(pair) == 0:
                exclude.append(word)
                continue
            sm, lg = pair.split()
            if ( sm in chain(*[ d.values() for d in pluralables.values() ])
                 or lg in chain(*[ d.values() for d in pluralables.values() ]) ):
                exclude.append(word)
                continue
            if sm not in df.index or lg not in df.index:
                exclude.append(word)
                continue
            pluralables["neg_sm"][bin_] = sm
            pluralables["neg_lg"][bin_] = lg
            break

        # Get neutral words
        neu_candidates = [ w for w in sorted(dem_words_["term"],key = lambda t:
                           abs(dem_words_.loc[dem_words_["term"] == t]["valence"].values[0]
                           - np.median(dem_words_["valence"]))) if w not in exclude
                         ]
        for word in neu_candidates:
            pair = raw_input(word + " (NEUTRAL): ")
            if len(pair) == 0:
                exclude.append(word)
                continue
            sm, lg = pair.split()
            if ( sm in chain(*[ d.values() for d in pluralables.values() ])
                 or lg in chain(*[ d.values() for d in pluralables.values() ]) ):
                exclude.append(word)
                continue
            if sm not in df.index or lg not in df.index:
                exclude.append(word)
                continue
            pluralables["neu_sm"][bin_] = sm
            pluralables["neu_lg"][bin_] = lg
            break

    save_objs(pluralables, exclude)

def save_objs(pluralables, exclude):
    with open(INPUT_DIR + df_words_fn, "wb") as fh:
        pickle.dump(pd.DataFrame(pluralables), fh)

    with open(INPUT_DIR + exclude_fn, "w") as fh:
        fh.write("\n".join(exclude))

def words_to_csv():
    df_ = pickle.load(open(INPUT_DIR + df_words_fn, "rb"))

    pairs = []

    for cols in [["pos_sm", "pos_lg"], ["neg_sm", "neg_lg"],
                 ["neu_sm", "neu_lg"]]:
        pairs_ = df_[cols].values
        sm = [ pair[0] for pair in pairs_ ]
        lg = [ pair[1] for pair in pairs_ ]
        valence_sm = map(lambda w: get_valence(w)[0], sm)
        valence_lg = map(lambda w: get_valence(w)[0], lg)
        pairs += zip(sm, lg, valence_sm, valence_lg)

    sm = [ pair[0] for pair in magnitude_words ]
    lg = [ pair[1] for pair in magnitude_words ]
    valence_sm = map(lambda w: get_valence(w)[0], sm)
    valence_lg = map(lambda w: get_valence(w)[0], lg)
    pairs += zip(sm, lg, valence_sm, valence_lg)

    df__ = pd.DataFrame(pairs, columns = ["small", "large", "valence_sm",
                                          "valence_lg"])
    df__.to_csv(study_dir + "survey_materials/" + csv_words_fn)

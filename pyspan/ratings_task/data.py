from __future__ import division

import gensim.downloader as api
import itertools
import math
import matplotlib.pyplot as plt
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent = 4)
from scipy import stats
from pyspan.config import *
assert settings["mode"] == "crec"
INPUT_DIR = paths["input_dir"]
METRICS_DIR = paths["metrics_dir"]
TASK_PATH = paths["ratings_task_path"]
# Import a module written by Peter Norvig that interacts with data
# derived from the Google N-grams corpus
from pyspan.ngrams import ngrams
from pyspan.utils import *

# Response data
minidf = pd.read_csv(TASK_PATH + "responses.csv").set_index("index")

# Load relative frequency count data
partisan = pd.read_csv(TASK_PATH + "partisan.csv").set_index("index")
antonyms = pd.read_csv(TASK_PATH + "antonyms.csv").set_index("index")
all_ = pd.concat((partisan, antonyms))
prob_df = pickle.load(open(METRICS_DIR + "probs-unigrams", "rb"))
pkl_df = pickle.load(open(METRICS_DIR + "partial_kls-unigrams", "rb"))
pkl2_df = pickle.load(open(METRICS_DIR + "partial_kls2-unigrams", "rb"))
signal_df = pickle.load(open(METRICS_DIR + "signals-unigrams", "rb"))
logodds_df = pickle.load(open(METRICS_DIR + "logodds-unigrams", "rb"))
condprobs_df = pickle.load(open(METRICS_DIR + "cond_probs-unigrams", "rb"))
freq_df = pickle.load(open(METRICS_DIR + "frequencies-unigrams", "rb"))

# `p_ixs` contains indices corresponding to the partisan words
# This first line runs the analysis using all partisan words.
p_ixs = range(1, 79)
# This line restricts the analysis to only those words that were also
# seen by P's in the first study.
#dup_words = open("duplicated_terms.txt", "r").read().split("\n")
#dup_words = [ w for w in dup_words if w.strip() ]
#p_ixs = [ i for i in range(1, 79) if df_.loc[0][str(i)] in dup_words  ]
# Restricts p_ixs to words that are in the corpus
p_ixs = [ ix for ix in p_ixs if partisan.loc[ix]["word"] in freq_df.index ]

# `a_ixs` contains indices corresponding to the antonyms
a_ixs = range(79, 99)

# Some helper methods
def get_frequencies(ix):
    word = partisan.loc[ix]["word"]
    dfreq, rfreq = freq_df.loc[word]["dmetric"], freq_df.loc[word]["rmetric"]
    return dfreq, rfreq

def get_pkls(ix):
    word = all_.loc[ix]["word"]
    dpkl, rpkl = pkl_df.loc[word]["dmetric"], pkl_df.loc[word]["rmetric"]
    assert dpkl * rpkl < 0
    return dpkl * -1 if dpkl > 0 else rpkl
    #return rpkl

def get_probs(ix):
    word = partisan.loc[ix]["word"]
    dp, rp = prob_df.loc[word]["dmetric"], prob_df.loc[word]["rmetric"]
    return dp, rp

def get_signals(ix):
    word = partisan.loc[ix]["word"]
    dlogp, rlogp = signal_df.loc[word]["dmetric"], signal_df.loc[word]["rmetric"]
    assert dlogp * rlogp < 0
    return dlogp * -1 if dlogp > 0 else rlogp

def get_logodds(ix):
    word = partisan.loc[ix]["word"]
    dlo, rlo = logodds_df.loc[word]["dmetric"], logodds_df.loc[word]["rmetric"]
    return rlo

def get_marg_odds(ix):
    word = partisan.loc[ix]["word"]
    dc, rc = logit_coef_df.loc[word]["dmetric"], logit_coef_df.loc[word]["rmetric"]
    #dmo, rmo = math.exp(dc), math.exp(rc)
    return rc

# API for the GloVe word vectors
def load_glove():
    return api.load("glove-wiki-gigaword-100")

# Get IVs
def get_demographics():
    # Ages
    ages = (minidf["age"] - 18)/10.
    ages = replace_nans(ages)

    # Gender
    gender = pd.get_dummies(minidf["gender"])[["F"]]

    # Party affiliation
    parties = pd.get_dummies(minidf["party"])[["Democrat", "Republican"]]

    # Political leanings
    political_leanings = minidf["political_leanings"]
    political_leanings = replace_nans(political_leanings)

    # Strength of party identity
    party_identity = minidf["party_identity"]
    party_identity = replace_nans(party_identity)

    # Level of political engagement
    political_engagement = minidf["political_engagement"]
    political_engagement = replace_nans(political_engagement)

    # Level of education
    education = minidf["education"]
    education = replace_nans(education)

    # If they voted in the last election
    voted = minidf["voted"]
    voted = replace_nans(voted)

    # % of friends and family who share their political affiliation
    political_bubble = minidf["political_bubble"]/100.
    political_bubble = replace_nans(political_bubble)

    return { "age": ages, "gender": gender, "party": parties,
             "political leanings": political_leanings,
             "party identity": party_identity,
             "political engagement": political_engagement,
             "education": education, "voted": voted,
             "political bubble": political_bubble }

n_d_utterances = sum(freq_df["dmetric"])
n_r_utterances = sum(freq_df["rmetric"])
n_utterances = n_d_utterances + n_r_utterances
marginal_probs = map(lambda ix: sum(get_frequencies(ix))/n_utterances,
                     p_ixs)
pkls = map(get_pkls, p_ixs)
z_pkls = stats.mstats.zscore(pkls)
signals = map(get_signals, p_ixs)
z_signals = stats.mstats.zscore(signals)
# P(P|w) = [P(w|P)P(P)]/P(w)
cond_probs = [ np.multiply(np.array(get_probs(ix)),
                           np.array([ n_d_utterances, n_r_utterances ]))/(marginal_probs[p_ixs.index(ix)] * n_utterances)
               for ix in p_ixs ]
assert all([ 0 < cp < 1 for cp in itertools.chain(*cond_probs) ])
cond_probs = [ -1*cp[0] if cp[0] > cp[1] else cp[1] for cp in
               cond_probs ]
z_cprobs = stats.mstats.zscore(cond_probs)

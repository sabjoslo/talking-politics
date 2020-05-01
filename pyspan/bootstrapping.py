"""In order to generate confidence intervals on the unigram-level metrics,
generate 100 re-samplings of the same size (n = 68,569 Democratic speeches and
n = 75,440 Republican speeches).
"""
from __future__ import division
from collections import defaultdict
from itertools import chain
import multiprocessing as mp
import math
from nltk import FreqDist
import numpy as np
import os
import pandas as pd
import pickle
import random
import string
from pyspan.count_words import *

N = 68569 + 75440
N_SAMPLES = 100
N_CPUS = mp.cpu_count() - 1

BASE_DIR = "/Users/sabinasloman/Box/LoP/bootstraps/"
DATA_DIR = "/Users/sabinasloman/Box/LoP/data/debates/dfs/"
SAMPLE_DIR = BASE_DIR + "samples/"
COUNTS_DIR = BASE_DIR + "counts/"
METRICS_DIR = BASE_DIR + "metrics/"
ID_FN = BASE_DIR + ".ids"
VOCAB_FN = BASE_DIR + ".vocab"
DISTS_FN = BASE_DIR + "distributions"
CONFINTS_FN = BASE_DIR + "signals-confints"

# Get files to sample from
files = map(lambda dir_: [ dir_[0] + "/" + f for f in dir_[2] if "_speech" in f
                         ], os.walk(DATA_DIR))
files = list(chain(*files))

def _check_args(args):
    assert len(args) == 2
    assert isinstance(args[0], basestring)
    assert isinstance(args[1], dict)

# Pass an iterable where the first item is a unique identifier for the filename
# where the samples are dumped, and the second item is the attribute dictionary
# for a relevant Bootstrap object.
def get_one_sample(args):
    _check_args(args)
    id_, attrs = args

    dem_fh = open(SAMPLE_DIR + "dem_speech-" + id_, "w")
    repub_fh = open(SAMPLE_DIR + "repub_speech-" + id_, "w")

    counter = 0
    while counter < attrs["n"]:
        f = np.random.choice(files)
        fh = dem_fh if f.split("/")[-1] == "dem_speech" else repub_fh
        speech = np.random.choice(open(f, "r").read().split("\n"))
        if not speech.strip():
            continue
        fh.write(speech + "\n")
        counter += 1

    dem_fh.close()
    repub_fh.close()

def count_one_sample(id_):
    dem_fh = open(SAMPLE_DIR + "dem_speech-" + id_, "r")
    repub_fh = open(SAMPLE_DIR + "repub_speech-" + id_, "r")
    wfh = open(COUNTS_DIR + "counts-" + id_, "w")
    wfh.write("party|phrase|count\n")

    dugrams = defaultdict(lambda: 0)
    rugrams = defaultdict(lambda: 0)
    dtokens = process(dem_fh.read())
    for token in dtokens:
        dugrams[token] += 1
    rtokens = process(repub_fh.read())
    for token in rtokens:
        rugrams[token] += 1
    dugrams, rugrams = prune_ngram_counts(dugrams, rugrams)

    wfh.write("\n".join([ "D|{}|{}".format(k, dugrams[k]) for k in
                          dugrams.keys() ]))
    wfh.write("\n")
    wfh.write("\n".join([ "R|{}|{}".format(k, rugrams[k]) for k in
                          rugrams.keys() ]))

    dem_fh.close()
    repub_fh.close()
    wfh.close()

def calc_metrics_for_one_sample(id_):
    # Identify the file with the relevant counts
    rfh = open(COUNTS_DIR + "counts-" + id_, "r")

    # Adapted from the BitCounter's method get_freq_dists
    d_freq_dist = FreqDist()
    r_freq_dist = FreqDist()
    # Skip header
    rfh.readline()
    line = rfh.readline()
    while line.strip():
        party, phrase, count = line.strip().split("|")
        assert party in ("D", "R")
        count = int(count)
        if party == "D":
            d_freq_dist[phrase] += count
        else:
            r_freq_dist[phrase] += count
        line = rfh.readline()
    vocab = list(set(d_freq_dist.keys()).union(set(r_freq_dist.keys())))
    # L1 smoothing
    for phrase in vocab:
        d_freq_dist[phrase] += 1
        r_freq_dist[phrase] += 1

    # Adapted from the BitCounter's method get_signal
    # N.B. If denom == "q" is passed to get_signal, get_signal *should* be
    # redundant to get_log_odds with a change of base
    signal = lambda pi, qi: math.log(pi/qi, 2)
    signals = []
    for phrase in vocab:
        ds = signal(d_freq_dist.freq(phrase), r_freq_dist.freq(phrase))
        rs = signal(r_freq_dist.freq(phrase), d_freq_dist.freq(phrase))
        signals.append((phrase, ds, rs))
    df = pd.DataFrame(signals)
    df.columns = [ "term", "dmetric", "rmetric" ]
    df.set_index("term", inplace = True)
    df.to_pickle(METRICS_DIR + "signals-unigrams-" + id_)

# Pass an iterable where the first item is a word, and the second item is a dict
# with an item containing a list of ids with key "ids".
def simul_metrics_for_one_word(args):
    _check_args(args)
    word, d = args
    ids = d["ids"]
    files = [ METRICS_DIR + "signals-unigrams-" + id_ for id_ in ids ]

    def get_simul_metric(f):
        df = pickle.load(open(f, "rb"))
        try:
            return df.loc[word]["dmetric"]
        except KeyError:
            return np.nan

    m = map(get_simul_metric, files)
    return sorted(m)

# Pass an iterable where the first item is a list of sorted simulated metrics,
# and the second is a confidence level between 0 and 1.
def confint_from_simul(args):
    assert len(args) == 2
    assert hasattr(args[0], "__iter__")
    assert 0 <= args[1] <= 1
    a, conf_level = args

    marg = (1 - conf_level)/2
    s = int(round(len(a) * marg))
    e = len(a) - s
    return (a[s], a[e-1])

class Bootstrap():
    def __init__(self, n = N, n_samples = N_SAMPLES, n_cpus = N_CPUS, ids = None,
                 reload_ = False, write_ids = True):
        self.n = n
        self.n_samples = n_samples
        self.n_cpus = n_cpus
        if not ids and ( reload_ or not os.path.exists(ID_FN) ):
            ids = [ "".join(np.random.choice(list(string.lowercase), 4)) for _
                    in range(self.n_samples) ]
        elif not reload_:
            assert not ids
            ids = open(ID_FN, "r").read().split()
        assert len(ids) == self.n_samples
        self.ids = ids
        # Save ids to file so they can be reloaded later
        if write_ids:
            with open(ID_FN, "w") as wfh:
                wfh.write(" ".join(self.ids))

    def resample(self):
        pool = mp.Pool(self.n_cpus)
        pool.map(get_one_sample, iterable = [ (id_, self.__dict__) for id_ in
                                               self.ids ])
        pool.close()

    def count_samples(self):
        pool = mp.Pool(self.n_cpus)
        pool.map(count_one_sample, iterable = self.ids)
        pool.close()

    def calc_metrics(self):
        pool = mp.Pool(self.n_cpus)
        pool.map(calc_metrics_for_one_sample, iterable = self.ids)
        pool.close()

    def get_vocab(self, reload_ = False, write = True):
        if not reload_ and ( os.path.exists(VOCAB_FN) ):
            vocab = open(VOCAB_FN, "r").read().split()
        else:
            vocab = set()
            for id_ in self.ids:
                df = pickle.load(open(METRICS_DIR + "signals-unigrams-" + id_,
                                    "rb"))
                vocab = vocab.union(set(df.index))
            vocab = list(vocab)
        if write:
            with open(VOCAB_FN, "w") as wfh:
                wfh.write(" ".join(vocab))
        self.vocab = vocab

    def get_distributions(self):
        if not hasattr(self, "vocab"):
            self.get_vocab()

        pool = mp.Pool(self.n_cpus)
        simuls = pool.map(simul_metrics_for_one_word, iterable = [ (word,
                                                                    { "ids":
                                                                      self.ids
                                                                    }) for word
                                                                   in self.vocab
                                                                 ])
        pool.close()

        # Save as DataFrame
        simuls = np.array(simuls)
        df = pd.DataFrame(zip(self.vocab, *simuls.T))
        df.columns = [ "word" ] + range(simuls.shape[1])
        df.set_index("word", inplace = True)
        df.to_pickle(DISTS_FN)

        return simuls

    def calc_confints(self, conf_level = .95, reload_dists = False):
        # Get a confidence interval at the specified level for each word in the
        # vocabulary
        if not DISTS_FN.split("/")[-1] in os.listdir(BASE_DIR) or reload_dists:
            simuls = self.get_distributions()
        else:
            simuls = pickle.load(open(DISTS_FN, "rb")).values

        if not hasattr(self, "vocab"):
            self.get_vocab()

        pool = mp.Pool(self.n_cpus)
        confints = pool.map(confint_from_simul, iterable = [ (simul, conf_level)
                                                             for simul in simuls
                                                           ])
        pool.close()
        confints = np.array(list(confints))
        assert confints.shape == (len(self.vocab), 2)

        # Save as DataFrame
        df = pd.DataFrame(zip(self.vocab, confints[:,0], confints[:,1]))
        df.columns = [ "word", "lbound", "ubound" ]
        df.set_index("word", inplace = True)
        df.to_pickle(CONFINTS_FN)

if __name__ == "__main__":
    strap = Bootstrap()
    strap.resample()
    strap.count_samples()
    strap.calc_metrics()
    strap.calc_confints()

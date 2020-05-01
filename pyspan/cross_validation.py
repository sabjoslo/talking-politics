#!/usr/bin/python2.7

"""Calculate the predictive power of our metrics in determining the identity of
a speaker.
"""

from __future__ import division
from collections import defaultdict
from itertools import chain
import logging
import json
import math
import numpy as np
# Set the seed for reproducible results
np.random.seed(0)
import os
import pandas as pd
import pickle
import random
from sklearn import linear_model
from scipy import stats
import string
import sys
from pyspan import bitcounters
from pyspan.config import *
from pyspan.count_words import process, prune_ngram_counts
from pyspan.utils import *

class CrossValidator(object):
    def __init__(self):
        # Make a temporary directory to store counts
        self.tmp_dirname = "tmp" + "".join(random.choice(string.lowercase) for _
                                           in range(4)) + "/"

    def load_corpora(self, corpus = "crec", holdout = 0, test = .1,
                     equalize = True, small_sample = False):
        assert corpus in ("debates", "crec")
        if corpus == "debates":
            proc_txt_dir = debate_paths["proc_txt_dir"]
        else:
            proc_txt_dir = crec_paths["proc_txt_dir"]

        assert 0 <= holdout + test <= 1
        train = 1 - (holdout + test)
        assert train + holdout + test == 1

        # Load files
        files = list(chain(*[ [ "{}/{}".format(dir_[0], f) for f in dir_[2] ]
                              for dir_ in os.walk(proc_txt_dir) ]))
        dem_files = [ f for f in files if "dem_speech" in f ]
        repub_files = [ f for f in files if "repub_speech" in f ]
        # Hack, because sampling ten different training/holdout pairs from the
        # CRec is taking way too long
        if small_sample:
            assert isinstance(small_sample, int)
            dem_files = np.random.choice(dem_files, size = small_sample,
                                         replace = False)
            repub_files = np.random.choice(repub_files, size = small_sample,
                                           replace = False)

        # Load content in files
        dem_content = list(chain(*[ open(f, "r").read().split("\n") for f in
                                    dem_files ]))
        dem_content = [ process(s) for s in dem_content if len(process(s)) > 0 ]
        repub_content = list(chain(*[ [ s for s in
                                        open(f, "r").read().split("\n") if
                                        s.strip() ] for f in repub_files ]))
        repub_content = [ process(s) for s in repub_content if len(process(s)) >
                          0 ]

        # Generate indices to correspond to the training, holdout and test sets
        d_train_ixs = np.random.choice(range(len(dem_content)),
                                       int(round(len(dem_content) * train, 0)),
                                       replace = False)
        r_train_ixs = np.random.choice(range(len(repub_content)),
                                       int(round(len(repub_content) * train, 0)),
                                       replace = False)
        if train == 1:
            d_holdout_ixs, r_holdout_ixs, d_test_ixs, r_test_ixs = [], [], [], []
        else:
            d_test_ixs = [ ix for ix in range(len(dem_content)) if ix not in
                           d_train_ixs ]
            d_holdout_ixs = np.random.choice(d_test_ixs,
                                             int(round(len(dem_content) * holdout,
                                                       0)), replace = False)
            d_test_ixs = [ ix for ix in d_test_ixs if ix not in d_holdout_ixs ]
            r_test_ixs = [ ix for ix in range(len(repub_content)) if ix not in
                           r_train_ixs ]
            r_holdout_ixs = np.random.choice(r_test_ixs,
                                             int(round(len(repub_content) * holdout,
                                                       0)), replace = False)
            r_test_ixs = [ ix for ix in r_test_ixs if ix not in r_holdout_ixs ]

        assert len(d_test_ixs) - int(round(len(dem_content) * test, 0)) <= 1
        assert len(r_test_ixs) - int(round(len(repub_content) * test, 0)) <= 1
        assert ( len(d_train_ixs) + len(d_holdout_ixs) + len(d_test_ixs)
                 == len(dem_content) )
        assert ( len(r_train_ixs) + len(r_holdout_ixs) + len(r_test_ixs)
                 == len(repub_content) )
        assert ( len((set(d_train_ixs).union(set(d_holdout_ixs))).union(set(d_test_ixs)))
                 == len(dem_content) )
        assert ( len((set(r_train_ixs).union(set(r_holdout_ixs))).union(set(r_test_ixs)))
                 == len(repub_content) )

        # Separate content into training, holdout and test sets
        train_dem_content, holdout_dem_content, test_dem_content = [], [], []
        for i, content in enumerate(dem_content):
            if i in d_train_ixs:
                train_dem_content.append(content)
            elif i in d_holdout_ixs:
                holdout_dem_content.append(content)
            elif i in d_test_ixs:
                test_dem_content.append(content)

        train_repub_content, holdout_repub_content, test_repub_content = [], [], []
        for i, content in enumerate(repub_content):
            if i in r_train_ixs:
                train_repub_content.append(content)
            elif i in r_holdout_ixs:
                holdout_repub_content.append(content)
            elif i in r_test_ixs:
                test_repub_content.append(content)

        logging.info("""Ns before bootstrapping:

D Train: {}
R Train: {}
D Holdout: {}
R Holdout: {}
D Test: {}
R Test: {}
""".format(len(train_dem_content), len(train_repub_content),
           len(holdout_dem_content), len(holdout_repub_content),
           len(test_dem_content), len(test_repub_content)))

        if equalize:
            # Equalize samples so a baseline accuracy of .5 is meaningful
            if len(holdout_dem_content) > len(holdout_repub_content):
                bss = np.random.choice(holdout_repub_content,
                      size = len(holdout_dem_content) - len(holdout_repub_content))
                holdout_repub_content = np.append(holdout_repub_content, bss)
            elif len(holdout_repub_content) > len(holdout_dem_content):
                bss = np.random.choice(holdout_dem_content,
                      size = len(holdout_repub_content) - len(holdout_dem_content))
                holdout_dem_content = np.append(holdout_dem_content, bss)
            assert len(holdout_repub_content) == len(holdout_dem_content)

            if len(test_dem_content) > len(test_repub_content):
                bss = np.random.choice(test_repub_content,
                      size = len(test_dem_content) - len(test_repub_content))
                test_repub_content = np.append(test_repub_content, bss)
            elif len(test_repub_content) > len(test_dem_content):
                bss = np.random.choice(test_dem_content,
                      size = len(test_repub_content) - len(test_dem_content))
                test_dem_content = np.append(test_dem_content, bss)
            assert len(test_repub_content) == len(test_dem_content)

            logging.info("""Ns after bootstrapping:

D Train: {}
R Train: {}
D Holdout: {}
R Holdout: {}
D Test: {}
R Test: {}
""".format(len(train_dem_content), len(train_repub_content),
               len(holdout_dem_content), len(holdout_repub_content),
               len(test_dem_content), len(test_repub_content)))

        # Training content should consist of lists of words, not sentences
        train_dem_content = list(chain(*train_dem_content))
        train_repub_content = list(chain(*train_repub_content))

        return ( train_dem_content, train_repub_content,
                 holdout_dem_content, holdout_repub_content,
                 test_dem_content, test_repub_content )

    def write_counts(self, dem_speech, repub_speech):
        dgrams, rgrams = defaultdict(lambda: 0), defaultdict(lambda: 0)
        for token in dem_speech:
            dgrams[token] += 1
        for token in repub_speech:
            rgrams[token] += 1
        dgrams, rgrams = prune_ngram_counts(dgrams, rgrams, tol = 0)
        os.system("mkdir -p " + self.tmp_dirname)
        with open(self.tmp_dirname + "unigrams-2012.txt", "w") as wfh: # Just
        # use any one of the years that the BitCounter will find
            wfh.write('\n'.join([ 'D|{}|{}'.format(k,dgrams[k]) for k in
                                  dgrams.keys() ]))
            wfh.write('\n')
            wfh.write('\n'.join([ 'R|{}|{}'.format(k,rgrams[k]) for k in
                                  rgrams.keys() ]))

    def get_metrics(self, metric = "partial_kls", percent = 1.):
        # Get metrics
        bitcounters.NGRAM_DIR = self.tmp_dirname
        bitcounter = bitcounters.BitCounter(mode = "unigrams")
        bitcounter.get_freq_dists()
        if metric == "cond_probs":
            metrics = bitcounter.get_conditional_probs(save = False)
        elif metric == "partial_kls":
            metrics = bitcounter.get_partial_kl(save = False)
        elif metric == "probs":
            metrics = bitcounter.get_probs(save = False)
        elif metric == "signals":
            metrics = bitcounter.get_signal(save = False)

        # Weed out the top x% of partisan words, if applicable
        if percent < 1.:
            dmetrics = metrics.sort_values("dmetric", ascending = False)[:int(percent * len(metrics))]
            rmetrics = metrics.sort_values("rmetric", ascending = False)[:int(percent * len(metrics))]
            metrics = pd.concat([ dmetrics, rmetrics ])
            metrics = metrics[~metrics.index.duplicated(keep = "first")]

        return metrics

    def make_word_level_preds(self, metrics, dem_speech, repub_speech):
        speech = np.append(dem_speech, repub_speech)

        def _get_pred(w):
            try:
                metrics_ = list(metrics.loc[w][["dmetric", "rmetric"]].values)
                return metrics_.index(max(metrics_))
            except KeyError:
                return np.nan

        ws = set(speech)
        metrics_ = dict(zip(ws, map(_get_pred, ws)))

        preds_ = np.array(map(lambda w: metrics_[w], speech))
        assert len(preds_) == len(dem_speech) + len(repub_speech)
        return preds_

    def get_sentence_scores(self, metrics, dem_speech, repub_speech,
                            f = np.prod):
        def _get_metric(w):
            try:
                metrics__ = list(metrics.loc[w][["dmetric", "rmetric"]].values)
                return metrics__
            except KeyError:
                return [np.nan, np.nan]

        ws = set(list(chain(*dem_speech)) + list(chain(*repub_speech)))
        metrics_ = dict(zip(ws, map(_get_metric, ws)))

        sent_scores = []
        for sentence in np.append(dem_speech, repub_speech):
            sent_metrics = np.array([ metrics_[w] for w in sentence ])
            dmetrics = sent_metrics[:,0]
            dmetrics = dmetrics[~np.isnan(dmetrics)]
            rmetrics = sent_metrics[:,1]
            rmetrics = rmetrics[~np.isnan(rmetrics)]
            dscore = f(dmetrics) if len(dmetrics) > 0 else np.nan
            rscore = f(rmetrics) if len(rmetrics) > 0 else np.nan
            sent_scores.append((dscore, rscore))

        return sent_scores

    # TODO: Incorporate changes here into get_preds
    def predict_sentence_id(self, row):
        dscore, rscore = list(row[[0,1]].values)

        # Strict equality sometimes leads to floating point "errors" (e.g.
        # cases where the gmeans are considered equal but the prods aren't)
        if (np.isnan(dscore) and np.isnan(rscore)) or (dscore == rscore):
            return np.nan
        elif np.isnan(dscore):
            return 1
        elif np.isnan(rscore):
            return 0
        else:
            return 0 if dscore > rscore else 1

    def get_preds(self, level = "word", metric = "partial_kls", percent = 1.,
                  **kwargs):
        metrics = self.get_metrics(metric = metric, percent = percent, **kwargs)

        if level == "word":
            return self.make_word_level_preds(metrics,
                                              list(chain(*self.dY)),
                                              list(chain(*self.rY)))
        elif level == "sentence":
            return self.make_sentence_level_preds(metrics, self.dY, self.rY,
                                                  **kwargs)

    def get_truth(self, level = "word"):
        if level == "word":
            return np.array([0] * len(list(chain(*self.dY)))
                            + [1] * len(list(chain(*self.rY))))
        elif level == "sentence":
            return np.array([0] * len(self.dY) + [1] * len(self.rY))

    def score(self, preds, truth):
        assert len(preds) == len(truth)
        if len(preds) == 0:
            return 0
        truth = truth[~np.isnan(preds)]
        preds = preds[~np.isnan(preds)]
        n = len(preds)
        ixs = range(n)
        ma = np.ma.masked_where(preds[ixs] != truth[ixs], ixs)
        return len(ma.compressed())/n, n, sum(truth)/n

    def cross_validate(self, level = "word", metric = "partial_kls",
                       percent = 1., **kwargs):
        # Calling self.get_preds(level = "sentence") will return an array of
        # predictions for all sentences in the test set based on the <percent>%
        # most partisan words according to <metric>. The code in this `if` block
        # instead uses metrics calculated from the entire training corpus to
        # predict the party identity of the speaker of the <percent>% most
        # partisan sentences.
        if level == "sentence":
            # Load metrics
            paths_ = crec_paths if self.training_corpus == "crec" else \
                     debate_paths
            dir_ = paths_["metrics_dir"]
            metrics = pickle.load(open("{}{}-unigrams".format(dir_, metric), "rb"))
            # Get partisan scores for all sentences in the test set
            sent_scores = self.get_sentence_scores(metrics, self.dY, self.rY,
                                                   **kwargs)
            sents = pd.DataFrame(sent_scores)
            assert len(sents) == len(self.dY) + len(self.rY)

            # Restrict sentences used for scoring to the top <percent>% of
            # Democratic sentences and the top <percent>% of Republican
            # sentences
            dsents = sents.sort_values(0, ascending = False)[:int(percent * len(sents))]
            rsents = sents.sort_values(1, ascending = False)[:int(percent * len(sents))]
            sents = pd.concat([ dsents, rsents ])
            sents = sents[~sents.index.duplicated(keep = "first")]
            assert len(sents) == len(set(dsents.index).union(set(rsents.index)))

            dixs = [ ix for ix in sents.index if ix < len(self.dY) ]
            rixs = [ ix for ix in sents.index if ix >= len(self.dY) ]

            # Make predictions based on the already-calculated scores
            preds = sents.apply(self.predict_sentence_id, axis = 1).values

            truth = list(map(lambda ix: 0 if ix in dixs else 1, sents.index))
            truth = np.array(truth)

            assert len(preds) == len(truth) == len(sents)

            if percent == 1.:
                assert len(preds) == len(truth) == len(self.dY) + len(self.rY)

        else:
            preds = self.get_preds(level = level, metric = metric,
                                   percent = percent, **kwargs)
            truth = self.get_truth(level = level)
            assert len(preds) == len(truth) == len(list(chain(*self.dY))) + len(list(chain(*self.rY)))

        return self.score(preds, truth)

    def cross_correlate(self, percent = 1., test_metric = "cond_probs",
                        training_metric = "partial_kls", **kwargs):
        self.write_counts(self.dX, self.rX)
        training_metrics = self.get_metrics(metric = training_metric,
                                            percent = percent, **kwargs)
        dem_speech_ = list(chain(*self.dY))
        repub_speech_ = list(chain(*self.rY))
        self.write_counts(dem_speech_, repub_speech_)
        test_metrics = self.get_metrics(metric = test_metric, **kwargs)
        test_metrics = test_metrics.loc[training_metrics.index]
        training_metrics = training_metrics.loc[~np.isnan(test_metrics["dmetric"])]
        test_metrics = test_metrics.loc[training_metrics.index]
        assert training_metrics.index.equals(test_metrics.index)
        return stats.pearsonr(training_metrics["dmetric"], test_metrics["dmetric"]), \
            stats.pearsonr(training_metrics["rmetric"], test_metrics["rmetric"]), \
            len(training_metrics.index)

    def benchmark(self, corpus = "crec", metric = "cond_probs"):
        dem_speech, repub_speech, _, __, ___, ____ = self.load_corpora(corpus,
                                                                       test = 0)
        self.write_counts(dem_speech, repub_speech)
        metrics = self.get_metrics(metric = metric)
        preds = self.make_word_level_preds(metrics, dem_speech, repub_speech)
        truth = np.array([0] * len(dem_speech) + [1] * len(repub_speech))
        return self.score(preds, truth)

    def clean_up(self):
        os.system("rm -r " + self.tmp_dirname)

# The default cut_off value is the proportion of words that are Republican; at
# this value, all words are included
def plot_(from_, to, level, cut_off = 0.5145517326427427, fn = None,
          label = "accuracy", save = True, words = []):
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(1)

    label = label.lower()
    Label = label[0].upper() + label[1:]

    if not fn:
        fn = ".{}_{}_{}".format(from_, to, level)

    results = json.load(open(paths["output_dir"] + fn + ".json", "rb"))
    x = np.arange(.05, 1.05, .05)
    cut_off_ix = len(x)
    if not isinstance(cut_off, type(None)):
        cut_off_ix = np.where(x >= cut_off)[0][0] + 1
    for k in results.keys():
        y, n, b_rate = results[k]["y"], results[k]["n"], results[k]["base_rate"]

        # It looks like the base rate contains the percentage of Republican
        # words, so transform it to contain the maximum of the percentage of
        # Republican or Democratic words
        b_rate = np.vectorize(lambda i: max(i, 1-i))(b_rate)

        if "y_hi" and "y_lo" in results[k]:
            y_hi, y_lo = results[k]["y_hi"], results[k]["y_lo"]
            yerr = np.array([[ y[i] - y_lo[i] for i in range(len(x)) ],
                             [ y_hi[i] - y[i] for i in range(len(x)) ]])
            ax.errorbar(x[:cut_off_ix], y[:cut_off_ix], yerr = yerr,
                        elinewidth = 3, capsize = 4, color = "k",
                        label = Label)
        else:
            ax.plot(x[:cut_off_ix], y[:cut_off_ix], color = "k", label = Label)
        ax.set_xlabel(r"$logodds$ quantiles")
        ax.set_ylabel("Accuracy")

        if len(words) == 0:
            n = np.array(n)
            #n_adj = (n - min(n))/(max(n) - min(n))*(max(y) - min(y)) + min(y)
            n_adj = n / (max(n) * 1.2)
            ax.plot(x[:cut_off_ix], n_adj[:cut_off_ix], label = r"$n$")
            ax.annotate(r"$n = {}$".format(n[0]), xy = (x[0], n_adj[0]))
            ax.annotate(r"$n = {}$".format(n[cut_off_ix-1]),
                         xy = (x[cut_off_ix-1], n_adj[cut_off_ix-1]))

            # For the paper, I want a line indicating the performance of the
            # na\"{i}ve learner. For the debates, I calculated that 50.8\% of the
            # words were Republican words, so as a hack, put a line at .508
            #ax.set_xlim(*ax.get_xlim())
            #ax.plot(ax.get_xlim(), (.508,.508), color = "k", linestyle = "--")
            ax.plot(x[:cut_off_ix], b_rate[:cut_off_ix], color = "k",
                    linestyle = "--")
            ax.set_xlim(*ax.get_xlim())
            ax.plot(ax.get_xlim(), (.5,.5), color = "k", linewidth = .5)
            
        plt.legend()
        #ax.set_title("{} from {} to {} using {} ({} level)".format(Label, from_,
        #                                                           to,
        #                                                           " and ".join(k.split(":")),
        #                                                           level))
        ax.set_ylim(0,1)

        # Used to create vertical lines that indicate the percentile of all the
        # words in the iterable argument words
        if len(words) > 0:
            metrics = pickle.load(open("{}{}-unigrams".format(paths["metrics_dir"],
                                                              k), "rb"))
            dmetrics = metrics.sort_values("dmetric", ascending = False)
            dmetrics["pos"] = np.arange(len(dmetrics))
            rmetrics = metrics.sort_values("rmetric", ascending = False)
            rmetrics["pos"] = np.arange(len(rmetrics))
            for word, accuracy, conf in words:
                word_dat = metrics.loc[word]
                word_v = word_dat[[ "dmetric", "rmetric" ]]
                word_k = np.argmax(word_v)
                word_v = np.max(word_v)
                assert word_dat[word_k] == word_v
                metrics_ = dmetrics if word_k == "dmetric" else rmetrics
                perc = metrics_.loc[(word, "pos")] / len(metrics_)
                ix1, ix2 = np.where(x < perc)[0], np.where(x > perc)[0][0]
                if len(ix1) > 0:
                    ix1 = ix1[-1]
                    x1, x2 = x[ix1], x[ix2]
                    # This hasn't been written for within-sample comparisons yet
                    y1, y2 = y[ix1], y[ix2]
                    y_interp = y1 + (y2 - y1) / (x2 - x1) * (perc - x1)
                else:
                    y_interp = y[0]
                color = "b" if word_k == "dmetric" else "r"
                ax.scatter([perc, perc], [y_interp, y_interp], color = color,
                           marker = "|")
                ax.scatter([perc, perc], [accuracy, accuracy], color = color,
                           s = 5, alpha = .2)
                ax.errorbar([perc, perc], [accuracy, accuracy], yerr = conf,
                            fmt = ".", ms = 0, color = color, alpha = .1)

        if save:
            savefn = "{}_{}".format(save, "_".join(k.split(":")))
            if not isinstance(save, basestring):
                savefn = "{}{}_{}_{}_{}_{}".format(paths["output_dir"], from_,
                                                   to, level,
                                                   "_".join(k.split(":")), label)
            print savefn
            plt.savefig(savefn)
        return f, ax

def cross_validate_at_word_level(from_, to):
    cv = CrossValidator()
    results = defaultdict(dict)
    test_metric = "cond_probs" # Only relevant for cross-correlation
    if from_ != to:
        # Don't create a holdout set, since here we're not really model
        # selecting, just quantifying the predictive accuracy of chosen metrics
        logging.info("Getting data")
        cv.dX, cv.rX, _, __, ___, ____ = cv.load_corpora(from_, holdout = 0,
                                                         test = 0)
        _, __, ___, ____, cv.dY, cv.rY = cv.load_corpora(to, holdout = 0,
                                                         test = 1)

        # Get counts
        cv.write_counts(cv.dX, cv.rX)

        for metric in ("partial_kls", "cond_probs", "probs", "signals"):
            x_ = np.arange(.05, 1.05, .05)
            y_ = []
            n_ = []
            base_rates = []
            for percent in x_:
                score, n, base_rate = cv.cross_validate(level = "word",
                                                        metric = metric,
                                                        percent = percent)
                y_.append(score)
                #dcorr, rcorr, n = cv.cross_correlate(percent = percent,
                #                                     test_metric = test_metric,
                #                                     training_metric = metric)
                #y_.append(rcorr[0])
                n_.append(n)
                base_rates.append(base_rate)
            results[metric]["y"] = y_
            results[metric]["n"] = n_
            results[metric]["base_rate"] = base_rates

    # If we're randomly partitioning the data into the training and holdout set,
    # do this multiple times to get confidence intervals on the estimates
    else:
        dem_training_speech, dem_test_speech, \
        repub_training_speech, repub_test_speech = [], [], [], []
        small_sample = (from_ == "crec")*50
        for n in range(10):
            # Again, don't create holdout sets
            logging.info("Getting dataset #{}".format(n))
            dtr, rtr, _, __, dt, rt = cv.load_corpora(corpus = from_,
                                                      small_sample = small_sample,
                                                      holdout = 0, test = .1)
            dem_training_speech.append(dtr)
            dem_test_speech.append(dt)
            repub_training_speech.append(rtr)
            repub_test_speech.append(rt)
        for metric in ("partial_kls", "cond_probs", "probs", "signals"):
            x_ = np.arange(.05, 1.05, .05)
            y_mu = []
            y_lo = []
            y_hi = []
            n_ = []
            base_rates = []
            for percent in x_:
                ys, ns, base_rates_ = [], [], []
                for dtr, dt, rtr, rt in zip(dem_training_speech, dem_test_speech,
                                            repub_training_speech, repub_test_speech):
                    cv.dX = dtr
                    cv.dY = dt
                    cv.rX = rtr
                    cv.rY = rt
                    cv.write_counts(dtr, rtr)
                    score, n, base_rate = cv.cross_validate(level = "word",
                                                            metric = metric,
                                                            percent = percent)
                    ys.append(score)
                    #dcorr, rcorr, n = cv.cross_correlate(percent = percent,
                    #                                     test_metric = test_metric,
                    #                                     training_metric = metric)
                    #ys.append(rcorr[0])
                    ns.append(n)
                    base_rates_.append(base_rate)
                y_mu.append(np.mean(ys))
                y_sorted = sorted(ys)
                y_lo.append(y_sorted[1])
                y_hi.append(y_sorted[-2])
                n_.append(np.mean(ns))
                base_rates.append(np.mean(base_rates_))
            results[metric]["y"] = y_mu
            results[metric]["y_lo"] = y_lo
            results[metric]["y_hi"] = y_hi
            results[metric]["n"] = n_
            results[metric]["base_rate"] = base_rates

    assert results["partial_kls"]["y"][-1] == results["signals"]["y"][-1] == results["probs"]["y"][-1]

    with open(paths["output_dir"] + ".{}_{}_word.json".format(from_, to), "w") as wfh:
        json.dump(results, wfh)

    cv.clean_up()

def cross_validate_at_sentence_level(from_, to, holdout = .8, test = .2,
                                     fn = None, **kwargs):
    cv = CrossValidator()
    logging.info("Getting data")
    cv.training_corpus, cv.test_corpus = from_, to
    dem_training_speech, dem_test_speech, \
    repub_training_speech, repub_test_speech = [], [], [], []
    results = defaultdict(dict)
    for _ in range(10):
        # Just look at holdout data, for now
        dtr, rtr, dh, rh, _, __ = cv.load_corpora(corpus = to,
                                                  holdout = holdout, test = test,
                                                  **kwargs)
        dem_training_speech.append(dtr)
        dem_test_speech.append(dh)
        repub_training_speech.append(rtr)
        repub_test_speech.append(rh)

    # A hacky geometric mean
    def _gmean(a):
        prod_ = np.prod(a)
        a_ = map(abs, a)
        gmean_ = stats.mstats.gmean(a_)
        return math.copysign(gmean_, prod_)

    for metric in ("cond_probs", "signals"):
        for f, flab in ((np.prod, "prod"), (_gmean, "geom_mean"),
                        (np.sum, "sum"), (np.mean, "mean")):
            x_ = np.arange(.05, 1.05, .05)
            y_mu = []
            y_lo = []
            y_hi = []
            n_ = []
            base_rates = []
            for percent in x_:
                ys, ns, base_rates_ = [], [], []
                for dtr, dh, rtr, rh in zip(dem_training_speech, dem_test_speech,
                                            repub_training_speech,
                                            repub_test_speech):
                    cv.dX, cv.rX, cv.dY, cv.rY, = dtr, rtr, dh, rh
                    assert len(cv.dX) == len(cv.rX) == 0
                    score, n, base_rate = cv.cross_validate(level = "sentence",
                                                            metric = metric,
                                                            percent = percent,
                                                            f = f)
                    ys.append(score)
                    ns.append(n)
                    base_rates_.append(base_rate)
                y_mu.append(np.mean(ys))
                y_sorted = sorted(ys)
                y_lo.append(y_sorted[1])
                y_hi.append(y_sorted[-2])
                n_.append(np.mean(ns))
                base_rates.append(np.mean(base_rates_))

            results["{}:{}".format(metric, flab)]["y"] = y_mu
            results["{}:{}".format(metric, flab)]["y_lo"] = y_lo
            results["{}:{}".format(metric, flab)]["y_hi"] = y_hi
            results["{}:{}".format(metric, flab)]["n"] = n_
            results["{}:{}".format(metric, flab)]["base_rate"] = base_rates

    assert results["cond_probs:sum"]["y"][-1] == results["cond_probs:mean"]["y"][-1]
    #This will not in general be true because of floating point errors (see
    #09_21_18.ipynb).
    #assert results["signals:prod"]["y"][-1] == results["signals:geom_mean"]["y"][-1]
    assert results["signals:sum"]["y"][-1] == results["signals:mean"]["y"][-1]

    if not fn:
        fn = ".{}_{}_sentence".format(from_, to)
    with open(paths["output_dir"] + fn + ".json", "w") as wfh:
        json.dump(results, wfh)

if __name__ == "__main__":
    assert sys.argv[1] in ("crec", "debates") # The corpus to use calculated
    # metrics from
    from_ = sys.argv[1]
    assert sys.argv[2] == "to"
    assert sys.argv[3] in ("crec", "debates") # The corpus to cross-validate
    # against
    to = sys.argv[3]
    try:
        level = sys.argv[4].lstrip("--level=")
    except IndexError:
        level = "word"

    startLog("{}_{}_{}".format(from_, to, level))

    if level == "word":
        cross_validate_at_word_level(from_ = from_, to = to)
    elif level == "sentence":
        cross_validate_at_sentence_level(from_ = from_, to = to)

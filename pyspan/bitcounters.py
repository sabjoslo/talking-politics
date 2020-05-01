from __future__ import division

import os
import math
from nltk import FreqDist
import numpy as np
import pandas as pd
import pickle
import re
from scipy import stats
import sys
from pyspan.bootstrapping import DISTS_FN
from pyspan.config import *
NGRAM_DIR = paths["ngram_dir"]
METRICS_DIR = paths["metrics_dir"]
from pyspan.ngrams import ngrams
from pyspan.valence import get_valence

class BitCounter(object):
    def __init__(self, mode):
        self.mode=mode
        self.regex=r'%s-[0-9]{4}.txt'%self.mode
        self.get_freq_dists()

    def get_freq_dists(self):
        print 'Counting word occurrences...'

        self.d_freq_dist=FreqDist()
        self.r_freq_dist=FreqDist()

        for entry in os.listdir(NGRAM_DIR):
            if isinstance(re.match(self.regex, entry),
                          type(None)):
                continue
            print 'Processing {}...'.format(entry)
            with open(NGRAM_DIR+entry,'r') as fh:
                # Skip header
                fh.readline()
                line=fh.readline()
                while line.strip():
                    party,phrase,count=line.strip().split('|')
                    if party not in ( 'D','R' ):
                        line=fh.readline()
                        continue
                    count=int(count)
                    if party=='D':
                        self.d_freq_dist[phrase]+=count
                    elif party=='R':
                        self.r_freq_dist[phrase]+=count
                    line=fh.readline()

        self.vocab=list(set(self.d_freq_dist.keys()).union(set(self.r_freq_dist.keys())))

        # L1 smoothing
        for phrase in self.vocab:
            self.d_freq_dist[phrase] += 1
            self.r_freq_dist[phrase] += 1

    def get_frequencies(self, save=True):
        print 'Getting frequencies...'
        frequencies=[]
        for phrase in self.vocab:
            frequencies.append((phrase, int(self.d_freq_dist[phrase]),
                                int(self.r_freq_dist[phrase])))
        df=pd.DataFrame(frequencies, columns=[ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR+"frequencies-"+self.mode)
        return df

    def get_partial_kl(self, denom = "q", pfun = None, save = True):
        assert denom in ("mixture", "q")
        if save:
            assert denom == "q"
            save = "partial_kls" if save == True else save

        partial_kl_mixture=lambda p,pi,qi:p*math.log(2*pi/(pi+qi),2)
        partial_kl_q=lambda p,pi,qi:p*math.log(pi/qi,2)
        partial_kl = partial_kl_q if denom == "q" else partial_kl_mixture

        print 'Computing partial KLs...'
        pkls=[]
        for phrase in self.vocab:
            dp = self.d_freq_dist.freq(phrase)
            rp = self.r_freq_dist.freq(phrase)
            dscale = dp if not pfun else pfun(phrase)
            rscale = rp if not pfun else pfun(phrase)
            dpkl = partial_kl(dscale, dp, rp)
            rpkl = partial_kl(rscale, rp, dp)
            pkls.append((phrase, dpkl, rpkl))
        df=pd.DataFrame(pkls, columns=[ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR + save + "-" + self.mode)
        return df

    def get_signal(self, denom = "q", save=True):
        assert denom in ("mixture", "q")
        if save:
            assert denom == "q"

        signal_mixture = lambda pi, qi: math.log(2*pi/(pi+qi),2)
        signal_q = lambda pi, qi: math.log(pi/qi, 2)
        signal = signal_q if denom == "q" else signal_mixture

        print 'Computing signal reliability...'
        signals=[]
        for phrase in self.vocab:
            dsr = signal(self.d_freq_dist.freq(phrase),
                         self.r_freq_dist.freq(phrase))
            rsr = signal(self.r_freq_dist.freq(phrase),
                         self.d_freq_dist.freq(phrase))
            signals.append((phrase, dsr, rsr))
        df=pd.DataFrame(signals, columns=[ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR+"signals-"+self.mode)
        return df

    def get_logps(self, save=True):
        logp=lambda pi:math.log(pi, 2)

        print 'Computing log p\'s...'
        logps=[]
        for phrase in self.vocab:
            dlp = logp(self.d_freq_dist.freq(phrase))
            rlp = logp(self.r_freq_dist.freq(phrase))
            logps.append((phrase, dlp, rlp))
        df=pd.DataFrame(logps, columns=[ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR+"logps-"+self.mode)
        return df

    def get_mixtures(self, save=True):
        print 'Computing mixtures...'
        ms=[]
        for phrase in self.vocab:
            m = (self.d_freq_dist.freq(phrase) + self.r_freq_dist.freq(phrase))/2
            ms.append((phrase, m, m))
        df=pd.DataFrame(ms, columns=[ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR+"ms-"+self.mode)
        return df

    def get_probs(self, save=True):
        print 'Computing raw probabilities...'
        probs=[]
        for phrase in self.vocab:
            probs.append((phrase, self.d_freq_dist.freq(phrase),
                          self.r_freq_dist.freq(phrase)))
        df=pd.DataFrame(probs, columns=[ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR+"probs-"+self.mode)
        return df

    # TODO: Get rid of this function; I think it's redundant with get_signal().
    def get_log_odds(self, save=True):
        print 'Computing log odds...'
        logodds = []
        for phrase in self.vocab:
            logodds.append((phrase,
                            math.log(self.d_freq_dist.freq(phrase)/self.r_freq_dist.freq(phrase)),
                            math.log(self.r_freq_dist.freq(phrase)/self.d_freq_dist.freq(phrase))))
        df = pd.DataFrame(logodds, columns = [ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR + "logodds-" + self.mode)
        return df

    def get_conditional_probs(self, save=True):
        print 'Computing conditional probabilities...'
        cond_probs = []
        n_d, n_r = sum(self.d_freq_dist.values()), \
                   sum(self.r_freq_dist.values())
        n = n_d + n_r
        for phrase in self.vocab:
            marg_prob = ( self.d_freq_dist[phrase] + self.r_freq_dist[phrase] )/n
            cp_d, cp_r = np.multiply(np.array([ self.d_freq_dist.freq(phrase),
                                                self.r_freq_dist.freq(phrase) ]),
                                     np.array([ n_d, n_r ])) / ( marg_prob * n )
            #cp_d, cp_r = np.multiply(np.array([ self.d_freq_dist.freq(phrase),
            #                                    self.r_freq_dist.freq(phrase) ]),
            #                         np.array([ 1, 1 ])) / ( marg_prob * 2 )
            cond_probs.append((phrase, cp_d, cp_r))
        df = pd.DataFrame(cond_probs, columns = [ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR + "cond_probs-" + self.mode)
        return df

    def get_likelihood_ratios(self, save=True):
        print 'Computing likelihood ratios...'
        lrs = []
        n_d, n_r = sum(self.d_freq_dist.values()), \
                   sum(self.r_freq_dist.values())
        n = n_d + n_r
        for phrase in self.vocab:
            dp, rp = self.d_freq_dist.freq(phrase), self.r_freq_dist.freq(phrase)
            lrs.append((phrase, dp/rp, rp/dp))
        df = pd.DataFrame(lrs, columns = [ "term", "dmetric", "rmetric" ]).set_index("term")
        df["dmetric_std"] = stats.mstats.zscore(df["dmetric"])
        df["rmetric_std"] = stats.mstats.zscore(df["rmetric"])
        if save:
            df.to_pickle(METRICS_DIR + "likelihood_ratios-" + self.mode)
        return df

    def _get_valence(self, phrase):
        v, a, d = get_valence(phrase)
        return (phrase, v, a, d)

    def get_valence(self, save=True):
        print "Getting valence..."
        vals = [ self._get_valence(phrase) for phrase in self.vocab ]
        df = pd.DataFrame(vals, columns = [ "term", "valence", "arousal",
                                            "dominance" ]).set_index("term")
        if save:
            df.to_pickle(METRICS_DIR + "valence-" + self.mode)
        return df

    def get_all(self, save = True):
        self.get_frequencies(save = save)
        self.get_partial_kl(save = "partial_kls" if save else save)
        self.get_signal(save = save)
        self.get_probs(save = save)
        self.get_log_odds(save = save)
        self.get_conditional_probs(save = save)

class CorrectedEstimator(BitCounter):
    def __init__(self, mode):
        assert mode == "unigrams"
        super(CorrectedEstimator, self).__init__(mode = mode)

    def get_signal(self, denom = "q", estimator = np.mean, save = True):
        bstrap = pickle.load(open(DISTS_FN, "rb"))
        emp = super(CorrectedEstimator, self).get_signal(denom = denom,
                                                         save = True)
        df = emp.join(bstrap)
        # DeDeo et al., 2013
        df = pd.concat([ 2 * df["dmetric"] - df[i] for i in bstrap.columns ],
                       axis = 1)
        f = lambda a: estimator(a[~np.isnan(a)])
        dmetric = np.apply_along_axis(f, 1, df)
        df["dmetric"] = dmetric

        if save:
            df.to_pickle(METRICS_DIR + "signals-corr")
        return df

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from pyspan.AFC.data import *

# Get empirical probability of occurrence of the two parties in the dataset
def empirical_marginal_probs():
    freq_df = pickle.load(open(paths["metrics_dir"] + "frequencies-unigrams",
                               "rb"))
    n_d = sum(freq_df["dmetric"])
    n_r = sum(freq_df["rmetric"])
    n = n_d + n_r
    pr_d = n_d / n
    pr_r = n_r / n
    return pr_d, pr_r

class PerceptualData(object):
    def __init__(self, ixs = p_ixs, **kwargs):
        self.ixs = ixs
        self.data = minidf
        for k, v in kwargs.items():
            self.data = self.data.loc[self.data[k] == v]
        pairs = all_.loc[ixs][["word1", "word2"]].values
        self.pairs = [ tuple(pair) for pair in pairs ]

    def get_discriminability_by_pair(self, plot = True):
        self.discrim_by_pair = []
        for ix in self.ixs:
            responses = []
            for i in self.data.index:
                response = self.data.loc[i][str(ix)]
                if isinstance(response, str):
                    assert response in all_.values[ix-1]
                    responses.append(response in
                                     truth[self.data.loc[i]["Condition"]])
            self.discrim_by_pair.append(np.mean(responses))
        self.discrim_by_pair = dict(zip(self.pairs, self.discrim_by_pair))

        if plot:
            plt.hist(self.discrim_by_pair.values(), bins = 30)
            plt.title("Histogram of item-level discriminabilities")
            plt.show()

from __future__ import division
import math
import numpy as np
import pickle
from scipy import stats
from pyspan.config import *
import sys
EPS = sys.float_info.epsilon

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    probs = pickle.load(open(paths["metrics_dir"] + "probs-unigrams", "rb"))

    print stats.ks_2samp(probs["dmetric"], probs["rmetric"])

    # Plot cumulative fraction plots of the two distributions
    # Take the log of the probabilities
    #d = map(math.log, probs["dmetric"].replace(0, EPS))
    #r = map(math.log, probs["rmetric"].replace(0, EPS))
    d = map(math.log, filter(lambda p: p > 0, probs["dmetric"]))
    r = map(math.log, filter(lambda p: p > 0, probs["rmetric"]))

    cf = lambda f, dist: len(filter(lambda p: p <= f, dist)) / len(dist)

    allprobs = d + r
    xs = np.linspace(min(allprobs), max(allprobs), 100)
    dcf = map(lambda f: cf(f, d), xs)
    rcf = map(lambda f: cf(f, r), xs)

    plt.plot(dcf, "b", label = "Dem")
    plt.plot(rcf, "r", label = "Repub")
    plt.xlabel(r"$Log(P)$")
    plt.ylabel(r"$Pr(log(p) < P)$")
    plt.title("Cumulative fraction plots of word probability distributions")
    plt.legend()
    plt.savefig(paths["output_dir"] + "cumulative_fraction_plot")

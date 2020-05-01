"""Generate a classification accuracy for each candidate metric.
"""

from __future__ import division
import pickle
import sys
from pyspan.config import *
metrics_dir = paths["metrics_dir"]

metric, mode = sys.argv[1:]

freqs = pickle.load(open("{}frequencies-{}".format(metrics_dir, mode), "rb"))
metrics = pickle.load(open("{}{}-{}".format(metrics_dir, metric, mode), "rb"))

right, wrong = 0, 0
for w in freqs.index:
    winner, loser = "rmetric", "dmetric"
    if metrics.loc[w]["dmetric"] > metrics.loc[w]["rmetric"]:
        winner, loser = loser, winner
    right += freqs.loc[w][winner]
    wrong += freqs.loc[w][loser]
print right / (right + wrong)

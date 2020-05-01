from __future__ import division
import pandas as pd
import pickle
from pyspan.config import *
mode = settings["mode"]
assert mode == "crec"

# Load pre-processed data
path = paths["afc_task_path"]
partisan = pd.read_csv(path + "partisan_words.csv").set_index("index")
antonyms_ = pd.read_csv(path + "antonyms.csv").set_index("index")
all_ = pd.concat([ partisan, antonyms_ ])
minidf = pd.read_csv(path + "responses.csv")

# "Want" switched partisanship
p_ixs = range(1, 89)
p_ixs.remove(14)

truth = dict()
truth["DEMOCRAT"] = list(partisan.loc[p_ixs]["word1"])
truth["REPUBLICAN"] = list(partisan.loc[p_ixs]["word2"])

partisan_pairs = zip(truth["DEMOCRAT"], truth["REPUBLICAN"])
diff_sqs = dict(zip(partisan_pairs, partisan.loc[p_ixs]["DIFF_SQ"]))

antonyms = {
    "POSITIVE": [ "superior", "joy", "plentiful", "qualified", "laugh",
                  "clever", "rapid", "famous", "useful", "loyal" ],
    "NEGATIVE": [ "inferior", "sorrow", "scarce", "unqualified", "cry",
                  "stupid", "slow", "unknown", "useless", "disloyal" ]
}

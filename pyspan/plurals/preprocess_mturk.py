from __future__ import division
import os
import numpy as np
import pandas as pd
import pickle
import re
from pyspan.utils import *
from pyspan.config import *

BASE_DIR = paths["plurals_task_path"]

# Exclude participants who failed the attention check?
exclude = False
exclude = True
print("NOT "*(1 - exclude) + "EXCLUDING participants who failed the attention check from MTurk data.")

# Load data
words = pd.read_csv("{}stims_round2.csv".format(BASE_DIR)).set_index("Unnamed: 0")
# Reorder words to be consistent with dataframe
words_reordered = [ "paycheck", "girl", "achievement", "relationship", "song",
                    "friendship", "victory", "miracle", "winner", "bonus",
                    "homicide", "deficiency", "killer", "funeral", "nightmare",
                    "rapist", "difficulty", "crime", "failure", "death",
                    "proposal", "site", "region", "instruction", "particle",
                    "predecessor", "affiliation", "referral", "pillar", "town",
                    "second", "yard", "centimeter", "teaspoon", "cup",
                    "megabyte", "hertz", "hundred", "small", "narrow" ]
ixs_ = [ words.loc[words.small == word].index[0] for word in words_reordered ]
words = words.loc[ixs_]
words_original = pd.read_csv("{}Plurals-survey_terms.csv".format(BASE_DIR))
cats = pickle.load(open("{}Plurals-stimuli".format(BASE_DIR), "rb"))
neg = cats.loc[cats.neg_sm.isin(words.small)][["neg_lg","neg_sm"]]
assert len(neg) == 10
neu = cats.loc[cats.neu_sm.isin(words.small)][["neu_lg","neu_sm"]]
assert len(neu) == 10
pos = cats.loc[cats.pos_sm.isin(words.small)][["pos_lg","pos_sm"]]
assert len(pos) == 10
cats = pd.concat((neg, neu, pos), axis = 1)
neg_lg = cats.neg_lg[~is_nan(cats.neg_lg)].values
neg_sm = cats.neg_sm[~is_nan(cats.neg_sm)].values
neu_lg = cats.neu_lg[~is_nan(cats.neu_lg)].values
neu_sm = cats.neu_sm[~is_nan(cats.neu_sm)].values
pos_lg = cats.pos_lg[~is_nan(cats.pos_lg)].values
pos_sm = cats.pos_sm[~is_nan(cats.pos_sm)].values

#raw = pd.read_csv("{}mturk_data_with_order.csv".format(BASE_DIR))
#raw = raw.loc[raw.bonus == "1.00"]
#assert len(raw) == 300
#assert np.all(raw.Finished)
#raw["ID"] = np.arange(300)
#raw.to_csv("mturk_accepted.csv")

raw = pd.read_csv("{}mturk/mturk_accepted.csv".format(BASE_DIR))

# Extract order
order = raw["FL_32_DO"]
order = np.array([ order_.split("|") for order_ in order ])
assert order.shape == (300,3)
valence_order = np.array([ np.where(order_ == "FL_42")[0]+1 for order_ in order ])
assert valence_order.shape == (300,1)
valence_order = valence_order[:,0]
cl_order = np.array([ np.where(order_ == "FL_40")[0]+1 for order_ in order ])
assert cl_order.shape == (300,1)
cl_order = cl_order[:,0]
politics_order = np.array([ np.where(order_ == "FL_41")[0]+1 for order_ in order ])
assert politics_order.shape == (300,1)
politics_order = politics_order[:,0]
assert not np.any(valence_order == cl_order)
assert not np.any(valence_order == politics_order)
assert not np.any(cl_order == politics_order)
raw["valence_order"] = valence_order
raw["cl_order"] = cl_order
raw["politics_order"] = politics_order

valence_cols = map(lambda i: "Q{}".format(i), range(1545, 1555) +
                   range(1556, 1566) + range(1567, 1577) + range(1578, 1588))
valence_cols += map(lambda i: "Q{}".format(i), range(1633, 1643) +
                    range(1644, 1654) + range(1655, 1665) + range(1666, 1676))
valence_raw = raw[["ID", "valence_condition", "valence_order"] + valence_cols]
valence_raw.columns = np.append(["ID", "Condition", "order"],
                                np.append((words.index + 100).values,
                                          (words.index + 200).values))

cl_cols = map(lambda i: "Q{}".format(i), range(851, 861) + range(862, 872) +
              range(873, 883) + range(884, 894))
cl_cols += map(lambda i: "Q{}".format(i), range(1493, 1503) + range(1504, 1514)
               + range(1515, 1525) + range(1526, 1536))
cl_raw = raw[["ID", "construal_condition", "cl_order"] + cl_cols]
cl_raw.columns = np.append(["ID", "Condition", "order"],
                           np.append((words.index + 100).values,
                                     (words.index + 200).values))

politics_cols = map(lambda i: "Q{}".format(i), range(1589, 1599) +
                    range(1600, 1610) + range(1611, 1621) + range(1622, 1632))
politics_cols += map(lambda i: "Q{}".format(i), range(1677, 1687) +
                     range(1688, 1698) + range(1699, 1709) + range(1710, 1720))
politics_raw = raw[["ID", "political_condition", "politics_order"] +
                   politics_cols]
politics_raw.columns = np.append(["ID", "Condition", "order"],
                                 np.append((words.index + 100).values,
                                           (words.index + 200).values))

atc1_cols = map(lambda i: "Q{}".format(i), range(607, 707))
atc1_cols += map(lambda i: "Q{}".format(i), range(748, 848))
atc1 = raw[["ID"] + atc1_cols]
atc1.columns = np.append(["ID"], np.arange(100, 300))

atc2_cols = map(lambda i: "Q{}".format(i), range(895, 995))
atc2_cols += map(lambda i: "Q{}".format(i), range(996, 1096))
atc2 = raw[["ID"] + atc2_cols]
atc2.columns = np.append(["ID"], np.arange(100, 300))

demographics_cols = [ "Q1723", "Q1724", "Q1724_5_TEXT", "Q1725", "Q1726",
                      "Q1726_4_TEXT", "Q1727", "Q1728", "Q1729_1" ]
demographics_raw = raw[["ID"] + demographics_cols]
demographics_raw.columns = [ "ID", "age", "gender", "gender_5_TEXT", "Q13",
                             "Q4", "Q4_4_TEXT", "Q5", "Q9", "Q12_1" ]

if exclude:
    # Exclude data where the attention check was failed
    big_atc = np.in1d(atc1, words_original["large"]).reshape(atc1.shape)
    sm_atc = np.in1d(atc2, words_original["small"]).reshape(atc2.shape)
    atc = np.concatenate((big_atc, sm_atc), axis = 1)
    atc_failed = np.apply_along_axis(lambda a: sum(a) < 16, 1, atc)
    exclude = np.where(atc_failed)[0]

    valence_raw = valence_raw.loc[~np.in1d(valence_raw.ID, exclude)]
    cl_raw = cl_raw.loc[~np.in1d(cl_raw.ID, exclude)]
    politics_raw = politics_raw.loc[~np.in1d(politics_raw.ID, exclude)]
    demographics_raw = demographics_raw.loc[~np.in1d(demographics_raw.ID, exclude)]

    # How many people failed the attention check?
    # print (len(exclude) / 300)
    assert ( len(valence_raw) == len(cl_raw) == len(politics_raw) ==
             len(demographics_raw) == 300 - len(exclude) )

# Reformat data
# Valence
valence_raw.rename(columns = dict(zip(map(str,
                                          np.append((words.index + 100).values,
                                                    (words.index + 200).values)),
                                          np.append((words.index + 100).values,
                                                    (words.index + 200).values))),
                   inplace = True)
valence1 = valence_raw[(words.index + 100).values]
valence2 = valence_raw[(words.index + 200).values]
valence2.rename(columns = dict(zip((words.index + 200).values,
                                   (words.index + 100).values)), inplace = True)
valence = valence1.fillna(valence2)
valence[["Condition", "ID", "order"]] = valence_raw[["Condition", "ID", "order"]]
valence["Condition"] = valence["Condition"].replace({ "unhappy, annoyed, unsatisfied, melancholic, despaired, or bored": "NEGATIVE",
                                                      "happy, pleased, satisfied, contented, or hopeful": "POSITIVE" })
valence.set_index("ID", inplace = True)
# Check indexing
def check_indexing(ix, df):
    assert np.all(np.in1d(df[100+ix],
                  np.append("-99", words.loc[ix][["small","large"]])))
check_indexing = np.vectorize(check_indexing, excluded = [1])
check_indexing(words.index, valence)

# Construal level
cl_raw.rename(columns = dict(zip(map(str, np.append((words.index + 100).values,
                                                    (words.index + 200).values)),
                                          np.append((words.index + 100).values,
                                                    (words.index + 200).values))),
                   inplace = True)
cl1 = cl_raw[(words.index + 100).values]
cl2 = cl_raw[(words.index + 200).values]
cl2.rename(columns = dict(zip((words.index + 200).values,
                              (words.index + 100).values)), inplace = True)
cl = cl1.fillna(cl2)
cl[["Condition", "ID", "order"]] = cl_raw[["Condition", "ID", "order"]]
cl["Condition"] = cl["Condition"].replace({ "concrete, observable things or events": "CONCRETE",
                                            "abstract dispositions or characteristics of things or events": "ABSTRACT" })
cl.set_index("ID", inplace = True)
check_indexing(words.index, cl)

# Politics
politics_raw.rename(columns = dict(zip(map(str,
                                           np.append((words.index + 100).values,
                                                     (words.index + 200).values)),
                                           np.append((words.index + 100).values,
                                                     (words.index + 200).values))),
                    inplace = True)
politics1 = politics_raw[(words.index + 100).values]
politics2 = politics_raw[(words.index + 200).values]
politics2.rename(columns = dict(zip((words.index + 200).values,
                                    (words.index + 100).values)),
                 inplace = True)
politics = politics1.fillna(politics2)
politics[["Condition", "ID", "order"]] = politics_raw[["Condition", "ID", "order"]]
politics["Condition"] = politics["Condition"].replace({ "Republican": "REPUBLICAN",
                                                        "Democrat": "DEMOCRAT" })
politics.set_index("ID", inplace = True)
check_indexing(words.index, politics)

demographics = demographics_raw.set_index("ID")

# Join politics with demographic data
politics = politics.join(demographics["Q4"], how = "left")
politics.rename(columns = { "Q4": "ident" }, inplace = True)
politics["ident"] = politics["ident"].replace({ "Republican": "REPUBLICAN",
                                                "Democrat": "DEMOCRAT" })

# Recover magnitude stims.
mag_lg = [ word for word in words["large"] if word not in
           np.concatenate([ pos_lg, neu_lg, neg_lg ]) ]
assert len(mag_lg) == 10
mag_lg = np.array(mag_lg)
mag_sm = [ word for word in words["small"] if word not in
           np.concatenate([ pos_sm, neu_sm, neg_sm ]) ]
assert len(mag_sm) == 10
mag_sm = np.array(mag_sm)

def consensus_level(choices, words_ix, conditions, condition_labels):
    n = len(choices[choices != "-99"])
    npos = len(choices[(choices == words.loc[words_ix]["large"]) &
                       (conditions == condition_labels[0])])
    npos += len(choices[(choices == words.loc[words_ix]["small"]) &
                        (conditions == condition_labels[1])])
    return npos / n

# Get items for which there was general agreement that the pluralized form was
# more positively (negatively) valenced
def agreement_items_valence(threshold):
    positive, negative = [], []
    condition = valence["Condition"].values
    for i in words.index:
        choices = valence[i + 100]
        consensus_l = consensus_level(choices, i, condition,
                                      [ "POSITIVE", "NEGATIVE" ])
        if consensus_l > threshold:
            positive.append(words.loc[i]["large"])
            negative.append(words.loc[i]["small"])
        if consensus_l < 1 - threshold:
            negative.append(words.loc[i]["large"])
            positive.append(words.loc[i]["small"])

    assert len(positive) == len(negative)
    return positive, negative

# 60% item selection threshold
positive60, negative60 = agreement_items_valence(.6)
# 80% item selection threshold
positive80, negative80 = agreement_items_valence(.8)

# Get items for which there was general agreement that the pluralized form was
# more abstract (concrete)
def agreement_items_construal(threshold):
    abstract, concrete = [], []
    condition = cl["Condition"].values
    for i in words.index:
        choices = cl[i + 100]
        consensus_l = consensus_level(choices, i, condition,
                                      [ "ABSTRACT", "CONCRETE" ])
        if consensus_l > threshold:
            abstract.append(words.loc[i]["large"])
            concrete.append(words.loc[i]["small"])
        if consensus_l < 1 - threshold:
            concrete.append(words.loc[i]["large"])
            abstract.append(words.loc[i]["small"])

    assert len(abstract) == len(concrete)
    return abstract, concrete

abstract60, concrete60 = agreement_items_construal(.6)
abstract80, concrete80 = agreement_items_construal(.8)

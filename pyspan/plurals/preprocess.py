from __future__ import division
import os
import numpy as np
import pandas as pd
import pickle
import re
from pyspan.config import *

BASE_DIR = paths["plurals_task_path"]

# Exclude participants who failed the attention check?
#exclude = False
exclude = True
print("NOT "*(1 - exclude) + "EXCLUDING participants who failed the attention check from in-lab data.")

# Load data
words = pd.read_csv("{}Plurals-survey_terms.csv".format(BASE_DIR))
cats = pickle.load(open("{}Plurals-stimuli".format(BASE_DIR), "rb"))

valence_raw = pd.read_csv("{}in-lab/Valence.csv".format(BASE_DIR))
cl_raw = pd.read_csv("{}in-lab/Construal_level.csv".format(BASE_DIR))
politics_raw = pd.read_csv("{}in-lab/Politics.csv".format(BASE_DIR))
gender_raw = pd.read_csv("{}in-lab/Gender.csv".format(BASE_DIR))

valence_raw = valence_raw.loc[valence_raw.index[2:]]
valence_raw = valence_raw.loc[valence_raw["Finished"] == "TRUE"]
valence_raw.rename(columns = dict(zip(map(str, range(100, 700)), range(100, 700))),
                   inplace = True)
valence_raw["ID"] = map(lambda s: s.strip(), valence_raw["ID"])
valence_raw.reset_index(inplace = True)
cl_raw = cl_raw.loc[cl_raw.index[2:]]
cl_raw = cl_raw.loc[cl_raw["Finished"] == "TRUE"]
cl_raw["ID"] = map(lambda s: s.strip(), cl_raw["ID"])
cl_raw.rename(columns = dict(zip(map(str, range(100, 700)), range(100, 700))),
              inplace = True)
cl_raw.reset_index(inplace = True)
politics_raw = politics_raw.loc[politics_raw.index[2:]]
politics_raw = politics_raw.loc[politics_raw["Finished"] == "TRUE"]
politics_raw.rename(columns = dict(zip(map(str, range(100, 700)), range(100, 700))),
                    inplace = True)
politics_raw["ID"] = map(lambda s: s.strip(), politics_raw["ID"])
politics_raw.reset_index(inplace = True)
gender_raw = gender_raw.loc[gender_raw.index[2:]]
gender_raw = gender_raw.loc[gender_raw["Finished"] == "TRUE"]
gender_raw.rename(columns = dict(zip(map(str, range(100, 700)), range(100, 700))),
                  inplace = True)
gender_raw["ID"] = map(lambda s: s.strip(), gender_raw["ID"])
gender_raw.reset_index(inplace = True)

isdigit = np.vectorize(lambda i: i.isdigit())

meta_raw = pd.read_csv("{}in-lab/Meta.csv".format(BASE_DIR))
meta_raw = meta_raw.replace({ np.nan: "" })
meta_raw = meta_raw.loc[((meta_raw["DO-BR-FL_11"] != "") | (meta_raw["DO-BR-FL_31"] != "")
                         | (meta_raw["DO-BR-FL_51"] != "") |
                         (meta_raw["DO-BR-FL_64"] != "")) & (isdigit(meta_raw["ID"])) ]

demographics_raw = pd.read_csv("{}in-lab/Demographics.csv".format(BASE_DIR))
demographics = demographics_raw[["age", "gender", "gender_5_TEXT", "Q13", "Q4", "Q4_4_TEXT", "Q5", "Q9", "Q12_1",
                                 "ID"]]
demographics.drop_duplicates(inplace = True)
demographics = demographics.replace({ np.nan: "" })
demographics = demographics.loc[isdigit(demographics["ID"])]
demographics.set_index("ID", inplace = True)

# Exclude participants who took any survey more than once
dup_ids = [ i for i in valence_raw["ID"] if list(valence_raw["ID"]).count(i) > 1 ]
dup_ids += [ i for i in cl_raw["ID"] if list(cl_raw["ID"]).count(i) > 1 ]
dup_ids += [ i for i in politics_raw["ID"] if list(politics_raw["ID"]).count(i) > 1 ]
dup_ids += [ i for i in gender_raw["ID"] if list(gender_raw["ID"]).count(i) > 1 ]
dup_ids += [ i for i in meta_raw["ID"] if list(meta_raw["ID"]).count(i) > 1 ]

valence_raw = valence_raw.loc[~valence_raw["ID"].isin(dup_ids)]
cl_raw = cl_raw.loc[~cl_raw["ID"].isin(dup_ids)]
politics_raw = politics_raw.loc[~politics_raw["ID"].isin(dup_ids)]
gender_raw = gender_raw.loc[~gender_raw["ID"].isin(dup_ids)]
meta_raw = meta_raw.loc[~meta_raw["ID"].isin(dup_ids)]

assert len(valence_raw["ID"]) == len(np.unique(valence_raw["ID"]))
assert len(cl_raw["ID"]) == len(np.unique(cl_raw["ID"]))
assert len(politics_raw["ID"]) == len(np.unique(politics_raw["ID"]))
assert len(gender_raw["ID"]) == len(np.unique(gender_raw["ID"]))
assert len(meta_raw["ID"]) == len(np.unique(meta_raw["ID"]))

if exclude:
    # Exclude data where the attention check was failed
    big_ixs = range(300, 500)
    sm_ixs = range(500, 700)

    # Valence
    # First attention check
    valence_big = valence_raw[big_ixs].astype(str)
    valence_big = valence_big.values[valence_big.values != "nan"]
    valence_big = valence_big.reshape(len(valence_raw) * 10)
    ma_big = np.ma.masked_where(np.in1d(valence_big, words["large"]), valence_big)
    assert all([ word in words["large"].values for word in valence_big[ma_big.mask] ])
    assert all([ word == "-99" or word in words["small"].values for word in
                 valence_big[~ma_big.mask] ])
    # Second attention check
    valence_sm = valence_raw[sm_ixs].astype(str)
    valence_sm = valence_sm.values[valence_sm.values != "nan"]
    valence_sm = valence_sm.reshape(len(valence_raw) * 10)
    ma_sm = np.ma.masked_where(np.in1d(valence_sm, words["small"]), valence_sm)
    assert all([ word in words["small"].values for word in valence_sm[ma_sm.mask] ])
    assert all([ word == "-99" or word in words["large"].values for word in
                 valence_sm[~ma_sm.mask] ])

    ma_big = ma_big.mask.reshape((len(valence_raw), 10))
    ma_sm = ma_sm.mask.reshape((len(valence_raw), 10))
    atc_dat = np.concatenate([ ma_big, ma_sm ], axis = 1)
    assert atc_dat.shape == ((len(valence_raw), 20))

    # How many people failed the attention check?
    nright = np.apply_along_axis(lambda a: len(a[a]), 1, atc_dat)
    assert len(nright) == len(valence_raw)
    failed = valence_raw.index[nright < 16]
    #print len(failed), len(failed) / len(valence_raw)
    valence_raw = valence_raw.loc[~np.in1d(valence_raw.index, failed)]
    assert len(valence_raw) == atc_dat.shape[0] - len(failed)
    valence_raw.reset_index(inplace = True)

    # Construal level
    # First attention check
    cl_big = cl_raw[big_ixs].astype(str)
    cl_big = cl_big.values[cl_big.values != "nan"]
    cl_big = cl_big.reshape(len(cl_raw) * 10)
    ma_big = np.ma.masked_where(np.in1d(cl_big, words["large"]), cl_big)
    assert all([ word in words["large"].values for word in cl_big[ma_big.mask] ])
    assert all([ word == "-99" or word in words["small"].values for word in
                 cl_big[~ma_big.mask] ])
    # Second attention check
    cl_sm = cl_raw[sm_ixs].astype(str)
    cl_sm = cl_sm.values[cl_sm.values != "nan"]
    cl_sm = cl_sm.reshape(len(cl_raw) * 10)
    ma_sm = np.ma.masked_where(np.in1d(cl_sm, words["small"]), cl_sm)
    assert all([ word in words["small"].values for word in cl_sm[ma_sm.mask] ])
    assert all([ word == "-99" or word in words["large"].values for word in
                 cl_sm[~ma_sm.mask] ])

    ma_big = ma_big.mask.reshape((len(cl_raw), 10))
    ma_sm = ma_sm.mask.reshape((len(cl_raw), 10))
    atc_dat = np.concatenate([ ma_big, ma_sm ], axis = 1)
    assert atc_dat.shape == ((len(cl_raw), 20))

    # How many people failed the attention check?
    nright = np.apply_along_axis(lambda a: len(a[a]), 1, atc_dat)
    assert len(nright) == len(cl_raw)
    failed = cl_raw.index[nright < 16]
    #print len(failed), len(failed) / len(cl_raw)
    cl_raw = cl_raw.loc[~np.in1d(cl_raw.index, failed)]
    assert len(cl_raw) == atc_dat.shape[0] - len(failed)
    cl_raw.reset_index(inplace = True)

    # Politics
    # First attention check
    politics_big = politics_raw[big_ixs].astype(str)
    politics_big = politics_big.values[politics_big.values != "nan"]
    politics_big = politics_big.reshape(len(politics_raw) * 10)
    ma_big = np.ma.masked_where(np.in1d(politics_big, words["large"]), politics_big)
    assert all([ word in words["large"].values for word in politics_big[ma_big.mask] ])
    assert all([ word == "-99" or word in words["small"].values for word in
                 politics_big[~ma_big.mask] ])
    # Second attention check
    politics_sm = politics_raw[sm_ixs].astype(str)
    politics_sm = politics_sm.values[politics_sm.values != "nan"]
    politics_sm = politics_sm.reshape(len(politics_raw) * 10)
    ma_sm = np.ma.masked_where(np.in1d(politics_sm, words["small"]), politics_sm)
    assert all([ word in words["small"].values for word in politics_sm[ma_sm.mask] ])
    assert all([ word == "-99" or word in words["large"].values for word in
                 politics_sm[~ma_sm.mask] ])

    ma_big = ma_big.mask.reshape((len(politics_raw), 10))
    ma_sm = ma_sm.mask.reshape((len(politics_raw), 10))
    atc_dat = np.concatenate([ ma_big, ma_sm ], axis = 1)
    assert atc_dat.shape == ((len(politics_raw), 20))

    # How many people failed the attention check?
    nright = np.apply_along_axis(lambda a: len(a[a]), 1, atc_dat)
    assert len(nright) == len(politics_raw)
    failed = politics_raw.index[nright < 16]
    #print len(failed), len(failed) / len(politics_raw)
    politics_raw = politics_raw.loc[~np.in1d(politics_raw.index, failed)]
    assert len(politics_raw) == atc_dat.shape[0] - len(failed)
    politics_raw.reset_index(inplace = True)

    # Gender
    # First attention check
    gender_big = gender_raw[big_ixs].astype(str)
    gender_big = gender_big.values[gender_big.values != "nan"]
    gender_big = gender_big.reshape(len(gender_raw) * 10)
    ma_big = np.ma.masked_where(np.in1d(gender_big, words["large"]), gender_big)
    assert all([ word in words["large"].values for word in gender_big[ma_big.mask] ])
    assert all([ word == "-99" or word in words["small"].values for word in
                 gender_big[~ma_big.mask] ])
    # Second attention check
    gender_sm = gender_raw[sm_ixs].astype(str)
    gender_sm = gender_sm.values[gender_sm.values != "nan"]
    gender_sm = gender_sm.reshape(len(gender_raw) * 10)
    ma_sm = np.ma.masked_where(np.in1d(gender_sm, words["small"]), gender_sm)
    assert all([ word in words["small"].values for word in gender_sm[ma_sm.mask] ])
    assert all([ word == "-99" or word in words["large"].values for word in
                 gender_sm[~ma_sm.mask] ])

    ma_big = ma_big.mask.reshape((len(gender_raw), 10))
    ma_sm = ma_sm.mask.reshape((len(gender_raw), 10))
    atc_dat = np.concatenate([ ma_big, ma_sm ], axis = 1)
    assert atc_dat.shape == ((len(gender_raw), 20))

    # How many people failed the attention check?
    nright = np.apply_along_axis(lambda a: len(a[a]), 1, atc_dat)
    assert len(nright) == len(gender_raw)
    failed = gender_raw.index[nright < 16]
    #print len(failed), len(failed) / len(gender_raw)
    gender_raw = gender_raw.loc[~np.in1d(gender_raw.index, failed)]
    assert len(gender_raw) == atc_dat.shape[0] - len(failed)
    gender_raw.reset_index(inplace = True)

# Reformat data
# Valence
valence1 = valence_raw[range(100, 200)]
valence2 = valence_raw[range(200, 300)]
valence2.rename(columns = dict(zip(range(200, 300), range(100, 200))),
                inplace = True)
valence = valence1.fillna(valence2)
valence[["Condition", "ID"]] = valence_raw[["Condition", "ID"]]
valence["Condition"] = valence["Condition"].replace({ "unhappy, annoyed, unsatisfied, melancholic, despaired, or bored": "NEGATIVE",
                                                      "happy, pleased, satisfied, contented, or hopeful": "POSITIVE" })
valence.set_index("ID", inplace = True)

# Construal level
cl1 = cl_raw[range(100, 200)]
cl2 = cl_raw[range(200, 300)]
cl2.rename(columns = dict(zip(range(200, 300), range(100, 200))),
           inplace = True)
cl = cl1.fillna(cl2)
cl[["Condition", "ID"]] = cl_raw[["Condition", "ID"]]
cl["Condition"] = cl["Condition"].replace({ "concrete, observable things or events": "CONCRETE",
                                            "abstract dispositions or characteristics of things or events": "ABSTRACT" })
cl.set_index("ID", inplace = True)

# Politics
politics1 = politics_raw[range(100, 200)]
politics2 = politics_raw[range(200, 300)]
politics2.rename(columns = dict(zip(range(200, 300), range(100, 200))),
                 inplace = True)
politics = politics1.fillna(politics2)
politics[["Condition", "ID"]] = politics_raw[["Condition", "ID"]]
politics.set_index("ID", inplace = True)

# Gender
gender1 = gender_raw[range(100, 200)]
gender2 = gender_raw[range(200, 300)]
gender2.rename(columns = dict(zip(range(200, 300), range(100, 200))),
               inplace = True)
gender = gender1.fillna(gender2)
gender[["Condition", "ID"]] = gender_raw[["Condition", "ID"]]
gender.set_index("ID", inplace = True)

# Extract order from Meta data
meta = meta_raw[["ID", "DO-BR-FL_11", "DO-BR-FL_31", "DO-BR-FL_51", "DO-BR-FL_64"]]

assert all([ len(re.findall("\|", val)) == 3 or val == "" for val in meta_raw["DO-BR-FL_11"] ])
assert all([ len(re.findall("\|", val)) == 2 or val == "" for val in meta_raw["DO-BR-FL_31"] ])
assert all([ len(re.findall("\|", val)) == 1 or val == "" for val in meta_raw["DO-BR-FL_51"] ])
assert all([ len(re.findall("\|", val)) == 0 for val in meta_raw["DO-BR-FL_64"] ])

get_flow_id = lambda l: l.split("|")[0]

meta["DO-BR-FL_11"] = [ get_flow_id(l) for l in meta["DO-BR-FL_11"].values ]
meta["DO-BR-FL_31"] = [ get_flow_id(l) for l in meta["DO-BR-FL_31"].values ]
meta["DO-BR-FL_51"] = [ get_flow_id(l) for l in meta["DO-BR-FL_51"].values ]
meta["DO-BR-FL_64"] = [ get_flow_id(l) for l in meta["DO-BR-FL_64"].values ]

order_dict = { "FL_14": "valence", "FL_32": "valence", "FL_52": "valence",
               "FL_65": "valence",
               "FL_16": "construal", "FL_35": "construal", "FL_55": "construal",
               "FL_68": "construal",
               "FL_21": "politics", "FL_38": "politics", "FL_58": "politics",
               "FL_71": "politics",
               "FL_19": "gender", "FL_41": "gender", "FL_61": "gender", "FL_74": "gender"
             }

meta = meta.replace(order_dict)
meta.columns = ["ID"] + range(1, 5)
meta.set_index("ID", inplace = True)

def get_order(id_, key):
    if meta.loc[id_][1] == key:
        return 1
    if meta.loc[id_][2] == key:
        return 2
    if meta.loc[id_][3] == key:
        return 3
    if meta.loc[id_][4] == key:
        return 4

get_order = np.vectorize(get_order, excluded = ["key"])

# Get order for valence.
valence["order"] = get_order(id_ = valence.index, key = "valence")

# Get order for construal level.
cl["order"] = get_order(id_ = cl.index, key = "construal")

# Get order for politics.
politics["order"] = get_order(id_ = politics.index, key = "politics")

# Get order for gender.
gender["order"] = get_order(id_ = gender.index, key = "gender")

# Join politics with demographic data and only keep observations from identified
# Democrats or Republicans
politics = politics.join(demographics["Q4"], how = "left")
# https://stackoverflow.com/questions/13035764#34297689
politics = politics[~politics.index.duplicated(keep = False)]
politics = politics.loc[politics["Q4"].isin([ "Republican", "Democrat" ])]
politics.rename(columns = { "Q4": "ident" }, inplace = True)
politics.replace({ "Republican": "REPUBLICAN", "Democrat": "DEMOCRAT" }, inplace = True)

# Join gender with demographic data and only keep observations from identified
# males or females
gender = gender.join(demographics["gender"], how = "left")
# https://stackoverflow.com/questions/13035764#34297689
gender = gender[~gender.index.duplicated(keep = False)]
gender = gender.loc[gender["gender"].isin(("Male", "Female"))]
gender.rename(columns = { "gender": "ident" }, inplace = True)
gender.replace({ "Male": "MALE", "Female": "FEMALE" }, inplace = True)

# Recover magnitude stims.
mag_lg = [ word for word in words["large"] if word not in np.concatenate([ cats["pos_lg"],
                                                                           cats["neu_lg"],
                                                                           cats["neg_lg"]
                                                                         ]) ]
assert len(mag_lg) == 25
mag_sm = [ word for word in words["small"] if word not in np.concatenate([ cats["pos_sm"],
                                                                           cats["neu_sm"],
                                                                           cats["neg_sm"]
                                                                         ]) ]
assert len(mag_sm) == 25

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

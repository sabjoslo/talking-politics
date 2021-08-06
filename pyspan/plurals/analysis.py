from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats
import sys
mturk = False
mturk = True
from pyspan.config import *
from pyspan.ratings_task.analysis import SparseLR

def load_participant_data(mturk):
    path = paths["package_dir"] + "experiments/valence/study-3"
    path += "'"*(not mturk)+"/"
    
    vars = dict()
    vars["words"] = pd.read_csv(path + "stimuli.csv").set_index("Unnamed: 0")
    vars["ixs"] = (vars["words"].index + 100).values
    vars["pos_lg"] = vars["words"].loc[vars["words"].category=="positive"].large
    vars["pos_sm"] = vars["words"].loc[vars["words"].category=="positive"].small
    vars["neg_lg"] = vars["words"].loc[vars["words"].category=="negative"].large
    vars["neg_sm"] = vars["words"].loc[vars["words"].category=="negative"].small
    vars["neu_lg"] = vars["words"].loc[vars["words"].category=="neutral"].large
    vars["neu_sm"] = vars["words"].loc[vars["words"].category=="neutral"].small
    
    vars["valence"] = pd.read_csv(path + "valence.csv").rename(columns=dict(zip(map(str,
                                                                                    vars["ixs"]
                                                                                   ), vars["ixs"]
                                                                               ))).set_index("ID")
    vars["cl"] = pd.read_csv(path + "cl.csv").rename(columns=dict(zip(map(str, vars["ixs"]),
                                                                      vars["ixs"]
                                                                     ))).set_index("ID")
    vars["politics"] = pd.read_csv(path + "politics.csv").rename(columns=dict(zip(map(str,
                                                                                  vars["ixs"]
                                                                                 ), vars["ixs"]
                                                                             ))).set_index("ID")
    vars["demographics"] = pd.read_csv(path + "demographics.csv").set_index("ID")
    
    if not mturk:
        vars["gender"] = pd.read_csv(path + "gender.csv").rename(columns=dict(zip(map(str,
                                                                                  vars["ixs"]
                                                                                 ), vars["ixs"]
                                                                             ))).set_index("ID")
        
    return vars
    
locals().update(load_participant_data(mturk))

def consensus_level(choices, words_ix, conditions, condition_labels):
    n = len(choices[choices != "-99"])
    npos = len(choices[(choices == words.loc[words_ix]["large"]) &
                       (conditions == condition_labels[0])])
    npos += len(choices[(choices == words.loc[words_ix]["small"]) &
                        (conditions == condition_labels[1])])
    return npos / n
    
# Get items for which there was general agreement about relative perceptions
# of the pluralized vs. singular forms
def agreement_items(df, attrs, threshold):
    attr1, attr2 = [], []
    condition = df["Condition"].values
    for i in words.index:
        choices = df[i + 100]
        consensus_l = consensus_level(choices, i, condition, attrs)
        if consensus_l > threshold:
            attr1.append(words.loc[i]["large"])
            attr2.append(words.loc[i]["small"])
        if consensus_l < 1 - threshold:
            attr2.append(words.loc[i]["large"])
            attr1.append(words.loc[i]["small"])

    assert len(attr1) == len(attr2)
    return attr1, attr2

# The t-tests are run on per-subject proportions, so collapse data into a
# proportion
def _get_proportions_within_subjects(arr):
    npos = len(arr[arr == 1])
    nneg = len(arr[arr == 0])
    if npos == nneg == 0:
        return np.nan
    return npos / (npos + nneg)

def summarize(df, ixs = ixs):
    summary = df[["Condition","ident"]]
    dat = df[ixs].values
    props = np.apply_along_axis(_get_proportions_within_subjects, 1, dat)
    summary["p"] = props
    assert summary.values.shape == (len(df), 3)
    return summary

def recode_within_subjects(df1, df2 = valence, conditions = [ "POSITIVE",
                                                              "NEGATIVE" ],
                           ixs = ixs):
    ixs = np.array(ixs)
    df1.drop(columns = [ i for i in range(100, 200) if i in df1.columns and i
                         not in ixs ], inplace = True)
    for i in df1.index:
        if i not in df2.index:
            df1 = df1.loc[df1.index != i]
            continue
        if df2.loc[i]["Condition"] == conditions[0]:
            l1 = df2.loc[(i,ixs)].values
            l2 = words.loc[((ixs-100)[l1 != "-99"], ["large","small"])].values
            l2 = np.ravel(l2)
            l2 = l2[~(np.in1d(l2, l1))]
        elif df2.loc[i]["Condition"] == conditions[1]:
            l2 = df2.loc[(i,ixs)].values
            l1 = words.loc[((ixs-100)[l2 != "-99"], ["large","small"])].values
            l1 = np.ravel(l1)
            l1 = l1[~(np.in1d(l1, l2))]
        l1 = l1[l1 != "-99"]
        l2 = l2[l2 != "-99"]
        assert len(l1) == len(l2)
        choices = df1.loc[(i,ixs)]
        choices_ = [ 1 if choice in l1 else choice for choice in choices ]
        choices_ = [ 0 if choice in l2 else choice for choice in choices_ ]
        choices_ = [ choice if choice in (0, 1) else np.nan for choice in choices_ ]
        df1.loc[(i,ixs)] = choices_
        df1.loc[(i, "order")] = int(df1.loc[(i, "order")] < df2.loc[(i, "order")])

    return df1

# Add columns with participant dummy variables, recoded so members of set1 are
# 1 (and members of set2 are 0), and observations of class1 are 1 (and of class2
# are 0)
#
# Inputs: df, a pandas DataFrame (one of or that has the same format as valence,
#       cl, politics and gender)
#   sets: a 2 x n numpy array, where row 0 contains the elements of set1 and row
#       1 contains the elements of set2. If within is True, this argument is
#       ignored.
#   classes: an iterable of size 2, where the first element is the class1 label
#       and the second element is the class2 label
def dummy(df, classes, sets = None, within = False, ixs = ixs, **kwargs):
    df_ = df.copy()
    if within:
        df_ = recode_within_subjects(df_, ixs = ixs, **kwargs)

    participant_dummies = pd.get_dummies(df_.index).set_index(df_.index)
    participant_dummies.columns = 1000 + participant_dummies.columns.astype(int)
    dummied = pd.concat([ df_, participant_dummies ], axis = 1)
    assert all([ (dummied.loc[i][1000 + int(i)] == 1).all() for i in df_.index ])
    # Remove one of the dummies to avoid multicollinearity
    dummied = dummied.drop(1000 + int(df_.index[0]), axis = 1)

    if not within:
        # TODO: Right now this doesn't check for ixs, and just assumes you're
        # using the whole df
        dummied.replace(sets[0,:], 1, inplace = True)
        dummied.replace(sets[1,:], 0, inplace = True)
        dummied.replace(np.ravel(words[["large", "small"]].values), np.nan,
                        inplace = True)
        dummied["order"] = dummied["order"] != 1

    dummied.replace(classes[0], 1, inplace = True)
    dummied.replace(classes[1], 0, inplace = True)
    dummied.replace("-99", np.nan, inplace = True)

    # Add signal data to dataframe
    choices = df_[ixs]
    signals = words.loc[ixs-100]["large_sig"] - words.loc[ixs-100]["small_sig"]
    signals = np.tile(signals, len(dummied)).reshape((len(dummied), len(ixs)))
    chose_pos_mask = dummied[ixs].values.astype(bool)
    chose_pl_mask = np.isin(choices.values, words.large)
    signals[(chose_pos_mask) & (~chose_pl_mask)] = -1 * signals[(chose_pos_mask) & (~chose_pl_mask)]
    signals[(~chose_pos_mask) & (chose_pl_mask)] = -1 * signals[(~chose_pos_mask) & (chose_pl_mask)]
    dummied[map(lambda i: "signal{}".format(i), ixs)] = pd.DataFrame(signals,
                                                                     index = dummied.index)

    # Reorder columns so the participant dummies are at the end
    # n columns should be n columns in original DataFrame + n participant
    # dummies + signal columns
    assert len(dummied.columns) == len(df_.columns)+(len(df_)-1)+len(ixs)
    cols = dummied.columns[:-(len(df_)-1+len(ixs))]
    cols = np.append(cols, dummied.columns[-len(ixs):])
    cols = np.append(cols, dummied.columns[len(df_.columns):-len(ixs)])
    assert np.array_equal(cols[:len(df_.columns)], df_.columns)
    assert np.array_equal(cols[len(df_.columns):(len(df_.columns)+len(ixs))],
                          np.array(map(lambda i: "signal{}".format(i), ixs)))
    assert np.array_equal(sorted(cols[(len(df_.columns)+len(ixs)):]),
                          sorted(df_.index[1:].astype(int)+1000))
    dummied = dummied[cols]

    return dummied, np.ravel(dummied[ixs])

def id_(dummied, ixs, *args):
    return np.repeat(dummied["ident"], len(ixs))

def condition_(dummied, ixs, *args):
    return np.repeat(dummied["Condition"], len(ixs))

def signal_(dummied, ixs, *args):
    return np.ravel(dummied[map(lambda i: "signal{}".format(i), ixs)])

def order_(dummied, ixs, *args):
    return np.repeat(dummied["order"], len(ixs))

def valence_(dummied, ixs, n):
    return np.tile(words.loc[(ixs-100,"valence_sm")]-5, n)

def df_to_matrix(dummied, Y, ixs = ixs,
                 columns = { 0: "id", 1: "condition", 2: (0,1), 3: "order" }):
    n, nunique = len(dummied), len(np.unique(dummied.index))
    nix = len(ixs)
    X = np.full((n * nix, len(columns) + nunique - 1), np.nan)
    fdict = { "id": id_, "condition": condition_, "signal": signal_,
              "order": order_, "valence": valence_ }
    for col, fid in columns.iteritems():
        if not isinstance(fid, basestring):
            assert hasattr(fid, "__iter__")
            assert isinstance(fid[0], int) and isinstance(fid[1], int)
            if len(fid) == 2:
                X[:,col] = X[:,fid[0]] == X[:,fid[1]]
                X[:,col][X[:,col] == 0] = -1
                continue
            elif len(fid) == 3:
                X[:,col] = fid[2](X[:,fid[0]], X[:,fid[1]])
                continue
            else:
                raise Exception
        X[:,col] = fdict[fid](dummied, ixs, n)
    for i in range(len(columns), len(columns) + nunique - 1):
        X[:,i] = np.repeat(dummied[dummied.columns[-(len(dummied)-1-i+len(columns))]], nix)

    X = X[~np.isnan(Y)]
    Y = Y[~np.isnan(Y)]
    Y = Y[~np.isnan(X).any(axis = 1)]
    X = X[~np.isnan(X).any(axis = 1),:]
    return X, Y

# Return the fraction of items in arr that belong to set1 (of those that
# belong to set1 or set2)
def get_prop(arr, set1, set2):
    n1 = np.count_nonzero(np.in1d(arr, set1))
    n2 = np.count_nonzero(np.in1d(arr, set2))
    assert n1 + n2 <= len(set1)

    try:
        return n1 / (n1 + n2)
    except ZeroDivisionError:
        return np.nan

def demographic_info(df):
    df_ = df.join(demographics[["age","gender"]], how = "left")
    assert len(df_.loc[df_.index.duplicated()]) == 0
    df_.age = pd.to_numeric(df_.age)
    print "Age: {} (SE = {})".format(np.mean(df_.age), stats.sem(df_.age))
    df_gp = df_.groupby("gender").count()
    print "Gender: {}".format(zip(df_gp.index, df_gp.values[:,0]))

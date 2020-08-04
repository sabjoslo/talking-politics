from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats
import sys
mturk = False
mturk = True
if not mturk:
    from pyspan.plurals.preprocess import *
    ixs = np.arange(100, 200)
else:
    from pyspan.plurals.preprocess_mturk import *
    ixs = (words.index + 100).values
from pyspan.ratings_task.analysis import SparseLR

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

# Infrastructure to generate bootstrapped p-values
def bootstrap_samples(summary_df, dat, lab, threshold = 60, within = False):
    ixs = [10, 20, 30] if mturk else [25, 50, 75]
    nwords = 40 if mturk else 100
    if lab == "valence":
        n_pos_missing = np.apply_along_axis(lambda a: len(a[a == "-99"]), 1,
                                            dat[:,:ixs[0]])
        n_pos = (nwords/4 - n_pos_missing).astype("int32")
        n_neg_missing = np.apply_along_axis(lambda a: len(a[a == "-99"]), 1,
                                            dat[:,ixs[0]:ixs[1]])
        n_neg = (nwords/4 - n_neg_missing).astype("int32")
        n_neu_missing = np.apply_along_axis(lambda a: len(a[a == "-99"]), 1,
                                            dat[:,ixs[1]:ixs[2]])
        n_neu = (nwords/4 - n_neu_missing).astype("int32")
    elif lab == "cl":
        n_missing = np.apply_along_axis(lambda a: len(a[a == "-99"]), 1, dat)
        n = nwords - n_missing
    elif lab in ("politics", "gender"):
        if within:
            n_missing = np.apply_along_axis(lambda a: len(a[~np.isnan(a)]), 1, dat)
            n = nwords - n_missing
        else:
            assert threshold in (60, 80)
            l = positive60 if threshold == 60 else positive80
            ixs_ = np.arange(len(words))[(np.in1d(words.small, l)) |
                                         (np.in1d(words.large, l))]
            n_missing = np.apply_along_axis(lambda a: len(a[a == "-99"]), 1,
                                            dat[:,ixs_])
            n = len(l) - n_missing

    for i in range(1000):
        dat_ = summary_df.copy()
        if lab == "valence":
            dat_["pos_lg"] = stats.binom.rvs(n = n_pos, p = .5) / n_pos
            dat_["neg_lg"] = stats.binom.rvs(n = n_neg, p = .5) / n_neg
            dat_["neu_lg"] = stats.binom.rvs(n = n_neu, p = .5) / n_neu
        elif lab == "cl":
            dat_["ppl"] = stats.binom.rvs(n = n, p = .5) / n
        elif lab in ("politics", "gender"):
            if within:
                dat_["p"] = stats.binom.rvs(n = n, p = .5) / n
            else:
                dat_["ppos"] = stats.binom.rvs(n = n, p = .5) / n

        flab = lab
        if lab in ("politics", "gender"):
            flab += "_ws" if within else str(threshold)
        with open("bts_samples{}/{}{}".format("_mturk" if mturk else "", flab, i),
                  "wb") as wfh:
            pickle.dump(dat_, wfh)

def _calculate_difference_in_means(i, col1, col2, cond1, cond2, flab, ident1,
                                   ident2, lab, recode):
    df = pickle.load(open("bts_samples{}/{}{}".format("_mturk" if mturk
                                                      else "", flab, i)))
    if recode:
        df1 = df.loc[df.Condition == recode[0]]
        df2 = df.loc[df.Condition == recode[1]]
        df2["pos_lg"] = 1 - df2["pos_lg"]
        df2["neg_lg"] = 1 - df2["neg_lg"]
        df2["neu_lg"] = 1 - df2["neu_lg"]
        df = pd.concat([ df1, df2 ])
    cols = ["Condition"]
    if lab in ("politics", "gender"):
        cols.append("ident")
    a = df[cols + [col1]]
    b = df[cols + [col2]]
    if not isinstance(cond1, type(None)):
        a = a.loc[a.Condition == cond1]
    if not isinstance(cond2, type(None)):
        b = b.loc[b.Condition == cond2]
    if not isinstance(ident1, type(None)):
        a = a.loc[a.ident == ident1]
    if not isinstance(ident2, type(None)):
        b = b.loc[b.ident == ident2]
    if lab in ("politics", "gender"):
        if ( isinstance(cond1, type(None)) and isinstance(cond2, type(None))
             and isinstance(ident1, type(None)) and
             isinstance(ident2, type(None)) ):
             a = a.loc[a.Condition == a.ident]
             b = b.loc[b.Condition != b.ident]

    a = a[col1].values
    b = b[col2].values
    return np.mean(a) - np.mean(b)

_calculate_difference_in_means = np.vectorize(_calculate_difference_in_means,
                                              excluded = range(1,10))

def calculate_p_value(delta, lab, cond1, col1, cond2, col2, ident1 = None,
                      ident2 = None, alternative = "smaller", recode = False,
                      threshold = 60, within = False):
    flab = lab
    if lab in ("politics", "gender"):
        flab += "_ws" if within else str(threshold)
    diff = _calculate_difference_in_means(range(1000), col1, col2, cond1, cond2,
                                          flab, ident1, ident2, lab, recode)
    if alternative == "smaller":
        return len([ d for d in diff if d < delta ]) / 1000
    elif alternative == "larger":
        return len([ d for d in diff if d > delta ]) / 1000

def demographic_info(df):
    df_ = df.join(demographics[["age","gender"]], how = "left")
    df_ = df_.loc[~df_.index.duplicated(keep = False)]
    df_.age = pd.to_numeric(df_.age)
    print "Age: {} (SE = {})".format(np.mean(df_.age), stats.sem(df_.age))
    df_gp = df_.groupby("gender").count()
    print "Gender: {}".format(zip(df_gp.index, df_gp.values[:,0]))

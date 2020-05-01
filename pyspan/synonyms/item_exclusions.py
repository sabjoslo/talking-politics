"""Implement the item-level exclusion criteria pre-registered for LoP_Synonyms:
https://osf.io/29yac
"""
from collections import defaultdict
from copy import deepcopy
from itertools import chain
import numpy as np
import pandas as pd
import pickle
import re
from scipy import stats
from pyspan import excerpt_sampler as excs
from pyspan.synonyms.data import *

excerpts = pickle.load(open("{}excerpts".format(TASK_PATH), "rb"))
original_excerpts = pickle.load(open("{}excerpts_short".format(TASK_PATH), "rb"))
syns = pd.read_csv("{}synonyms.csv".format(TASK_PATH))

remove_blanks = np.vectorize(lambda s: bool(s.strip()))
remove_whitespace = lambda s: re.sub(" +", " ", s).strip()

# Generate csvs for human annotators. Read from the "cleaned" versions of the
# sentences
def process_raw():
    csv_raw1 = pd.read_csv("{}excerpts_cleaned1.csv".format(TASK_PATH),
                           keep_default_na = False)
    original1 = csv_raw1["original"]
    updated1 = csv_raw1["updated/copied"]
    updated1 = updated1[remove_blanks(original1)]
    original1 = original1[remove_blanks(original1)]
    assert len(original1) == len(updated1) == 520

    csv_raw2 = pd.read_csv("{}excerpts_cleaned2.csv".format(TASK_PATH),
                           header = None, keep_default_na = False)
    original2 = csv_raw2[0]
    updated2 = csv_raw2[1]
    updated2 = updated2[remove_blanks(original2)]
    original2 = original2[remove_blanks(original2)]
    assert len(original2) == len(updated2) == 520

    original = np.append(original1, original2)
    updated = np.append(updated1, updated2)
    df = pd.DataFrame.from_records(zip(original, updated),
                                   columns = [ "original", "updated" ])
    excerpts_reversed = dict(chain(*[ [ (remove_whitespace(v),k) for v in
                                      excerpts[k] ] for k in excerpts.keys() ]))
    def id_word(s):
        try:
            return excerpts_reversed[remove_whitespace(s)]
        except KeyError:
            return "CAN'T_ID"
    id_word = np.vectorize(id_word)
    df["words"] = id_word(df["original"])

    # I'll manually replace the ones they didn't do, or in which replacements
    # of partial words (e.g. "terrorist" --> "fearist") occurred
    for i in df.index:
        word, original, s = df.loc[i][["words","original","updated"]]
        if not re.search(r"(?<!\w){}(?!\w)".format(word), s):
            df.loc[i]["updated"] = raw_input("""Word: {}
            Updated: {}
            Original: {}: """.format(word, s, original))
            continue
        if re.search(r"(\w{})|({}\w)".format(word, word), s):
            df.loc[i]["updated"] = raw_input("""Word: {}
            Updated: {}
            Original: {}: """.format(word, s, original))
            continue

    df.to_csv("{}excerpts_processed.csv".format(TASK_PATH))

# Replace sentences that were sampled twice
def replace_duplicates():
    to_sample = [ k for k in original_excerpts.keys() if
                  len(original_excerpts[k]) > len(set(original_excerpts[k])) ]
    sampler = excs.Sampler(to_sample)
    sampler.load_corpus()
    sampler.get_excerpts(n = 1, save = False)

    # Replace in saved pickle file
    original_excerpts_updated = deepcopy(original_excerpts)
    original_excerpts_updated[to_sample[0]] = list(set(original_excerpts_updated[to_sample[0]])) + [sampler.excerpts[to_sample[0]][0]]
    original_excerpts_updated[to_sample[1]] = list(set(original_excerpts_updated[to_sample[1]])) + [sampler.excerpts[to_sample[1]][0]]

    assert all([ len(np.unique(original_excerpts_updated[k])) == 10 for
                 k in original_excerpts_updated.keys() ])

    pickle.dump(original_excerpts_updated,
                open("{}excerpts_short_updated".format(TASK_PATH), "wb"))

    return [ v for v in original_excerpts[to_sample[0]] if
             list(original_excerpts[to_sample[0]]).count(v) > 1 ][0], \
           sampler.excerpts[to_sample[0]][0], [ v for v in
                                             original_excerpts[to_sample[1]] if
                                             list(original_excerpts[to_sample[1]]).count(v) > 1 ][0], \
           sampler.excerpts[to_sample[1]][0]

# Tag sentences as original vs. substitutions
def tag_sentences():
    df = pd.read_csv("{}excerpts_processed.csv".format(TASK_PATH))
    def index_string(s, w, ixs):
        s = remove_whitespace(s)
        excerpts_ = excerpts[w][ixs]
        excerpts_ = map(remove_whitespace, excerpts_)
        return excerpts_.index(s) if s in excerpts_ else -1000
    is_original = np.vectorize(lambda s,w: index_string(s, w, range(10)))
    is_substitution = np.vectorize(lambda s,w: index_string(s, w, range(10, 20)))
    original = is_original(df["original"], df["words"])
    substitution = is_substitution(df["original"], df["words"])
    assert all([ remove_whitespace(s) in map(remove_whitespace,
                                             original_excerpts[w]) for s, w
                 in df[["original","words"]].values[original > -1000] ])
    assert not any([ remove_whitespace(s) in map(remove_whitespace,
                                                 original_excerpts[w]) for
                     s, w in
                     df[["original","words"]].values[substitution > -1000] ])
    df["is_original"] = original
    df["is_substitution"] = substitution
    df.to_csv("{}excerpts_processed.csv".format(TASK_PATH))

def check_processed():
    df = pd.read_csv("{}excerpts_processed.csv".format(TASK_PATH))
    assert len(df.loc[df["is_original"] > -1000]) == len(df.loc[df["is_substitution"] > -1000]) == 520
    assert len(df.loc[(df["is_original"] > -1000) | (df["is_substitution"] > -1000)]) == len(df) == 1040
    assert len(np.unique(df["updated"])) == 1040
    for dw, rw in syns[["D","R"]].values:
        assert len(np.where(df["words"] == dw)[0]) == len(np.where(df["words"] == rw)[0]) == 20
        assert np.array_equal(np.array([-1000] * 10 + range(10)), np.sort(df.loc[df["words"] == dw]["is_original"]))
        assert np.array_equal(np.array([-1000] * 10 + range(10)), np.sort(df.loc[df["words"] == dw]["is_substitution"]))
        assert all([ re.search(r"(?<!\w){}(?!\w)".format(dw), s) for s in df.loc[df["words"] == dw]["updated"] ])
        assert all([ re.search(r"(?<!\w){}(?!\w)".format(rw), s) for s in df.loc[df["words"] == rw]["updated"] ])

# Divide the 1,040 sentences into two groups of 520 sentences, ensuring the
# original and substitution version of a sentence do not appear in the same
# group.
def regroup_sentences():
    df = pd.read_csv("{}excerpts_processed.csv".format(TASK_PATH))
    syns_pairs = np.concatenate((syns[["R","D"]].values,
                                 syns[["D","R"]].values), axis = 0)
    syns_dict = dict(syns_pairs)
    find_synonym = np.vectorize(lambda w: syns_dict[w])
    df["synonym"] = find_synonym(df["words"])
    df.to_csv("{}excerpts_processed.csv".format(TASK_PATH))
    l1, l2 = [], []
    tmp_df = deepcopy(df)
    while not tmp_df.empty:
        ix = np.random.choice(tmp_df.index)
        w, oix, six = tmp_df.loc[ix][["words","is_original","is_substitution"]].values
        match_ix = tmp_df.loc[(tmp_df["synonym"] == w) &
                              (tmp_df["is_original"] == six) &
                              (tmp_df["is_substitution"] == oix)].index
        assert len(match_ix) == 1
        match_ix = match_ix[0]
        l1.append(ix)
        l2.append(match_ix)
        tmp_df = tmp_df.loc[(tmp_df.index != ix) & (tmp_df.index != match_ix)]
    assert len(l1) == len(l2) == 520
    check = np.vectorize(lambda w1, w2, df_: len(df_.loc[((df_["words"] == w1) &
                         (df_["synonym"] == w2)) | ((df_["words"] == w2) &
                         (df_["synonym"] == w1))]), excluded = [2])
    group1 = df.loc[l1]
    assert np.all(check(syns_pairs[:,0], syns_pairs[:,1], group1) == 20)
    group2 = df.loc[l2]
    assert np.all(check(syns_pairs[:,0], syns_pairs[:,1], group2) == 20)
    group1["rating"] = np.nan
    group1 = group1[["updated","rating"]]
    group1.columns = ["sentence","rating"]
    group2["rating"] = np.nan
    group2 = group2[["updated","rating"]]
    group2.columns = ["sentence","rating"]
    group1.to_csv("{}group_1.csv".format(TASK_PATH))
    group2.to_csv("{}group_2.csv".format(TASK_PATH))

# For each item, we will have 40 sentences (20 per word), 20 of which are rated
# by the same two annotators (ratings 1 and 2; n = 20 each) and 20 of which are
# rated by the two other annotators (ratings 3 and 4; n = 20 each).
def items_to_exclude():
    excerpts = pd.read_csv("excerpts_processed.csv")
    excerpts.drop(columns = [ col for col in excerpts.columns if "Unnamed" in col ],
                  inplace = True)

    for fn, coln in zip(("annotations_1-1.csv", "annotations_1-2.csv",
                         "annotations_2-1.csv", "annotations_2-2.csv"),
                        ("rating_1", "rating_2", "rating_3", "rating_4")):
        annotations = pd.read_csv("item_level_exclusions/{}".format(fn))
        annotations.set_index("Unnamed: 0", inplace = True)
        excerpts = excerpts.join(annotations)
        _excerpts = excerpts.loc[[isinstance(s, basestring) for s in
                                  excerpts["sentence"]]]
        if not all([ remove_whitespace(u) == remove_whitespace(s) for u, s in
                     _excerpts[["updated","sentence"]].values ]):
            # Looks like one of the annotators slightly modified one of the
            # sentences
            assert coln == "rating_4"
            disagreements = np.where([ remove_whitespace(u) !=
                                       remove_whitespace(s) for u, s in
                                       _excerpts[["updated","sentence"]].values
                                     ])
            assert ( len(disagreements) == 1 and len(disagreements[0]) == 1 and
                     disagreements[0][0] == 360 )
        excerpts.drop(columns = [ "sentence" ], inplace = True)
        excerpts.rename(columns = { "rating": coln }, inplace = True)

    exclude = []

    for item in syns[["R","D"]].values:
        sentences = excerpts.loc[excerpts["words"].isin(item)]
        assert len(sentences) == 40

        # If the correlation between rating 1 and rating 2 is not greater than
        # 0, exclude the item from analysis.
        # Ratings 2 is missing one rating, so ignore that when calculating the
        # correlations
        if not np.array_equal(np.isnan(sentences["rating_1"]),
                              np.isnan(sentences["rating_2"])):
            assert item[0] == "illegal" and item[1] == "criminal"
            assert len(np.where(np.isnan(sentences["rating_1"]) !=
                       np.isnan(sentences["rating_2"]))) == 1
            assert sum(np.isnan(sentences["rating_1"])) == 20
            assert sum(np.isnan(sentences["rating_2"])) == 21
        if stats.pearsonr(sentences["rating_1"][~np.isnan(sentences["rating_2"])],
                          sentences["rating_2"][~np.isnan(sentences["rating_2"])]) <= 0:
            exclude.append(item)
            continue

        # If the correlation between rating 3 and rating 4 is not greater than
        # 0, exclude the item from analysis.
        # Ratings 3 is missing one rating, so ignore that when calculating the
        # correlations
        if not np.array_equal(np.isnan(sentences["rating_3"]),
                              np.isnan(sentences["rating_4"])):
            assert item[0] == "illegal" and item[1] == "criminal"
            assert len(np.where(np.isnan(sentences["rating_3"]) !=
                       np.isnan(sentences["rating_4"]))) == 1
            assert sum(np.isnan(sentences["rating_3"])) == 21
            assert sum(np.isnan(sentences["rating_4"])) == 20
        if stats.pearsonr(sentences["rating_3"][~np.isnan(sentences["rating_3"])],
                          sentences["rating_4"][~np.isnan(sentences["rating_3"])]) <= 0:
            exclude.append(item)
            continue

        # For each pair (original sentence, substitution sentence), calculate
        # the difference between the average rating of the substitution sentence
        # and the original sentence.
        original = sentences.loc[sentences["is_substitution"] == -1000].sort_values(by = ["is_original", "words"]).reset_index()
        substitution = sentences.loc[sentences["is_original"] == -1000].sort_values(by = ["is_substitution", "synonym"]).reset_index()
        assert np.array_equal(original["is_original"], substitution["is_substitution"])
        assert np.array_equal(original["words"], substitution["synonym"])

        avg_diffs = []
        for i in original.index:
            original_ratings = original.loc[(i,["rating_1","rating_2",
                                                "rating_3","rating_4"])].values
            original_ratings = np.array(original_ratings, dtype = float)
            original_ratings = original_ratings[~np.isnan(original_ratings)]
            if not len(original_ratings == 2):
                assert item[0] == "illegal" and item[1] == "criminal" and i == 0
                assert len(original_ratings) == 1
            original_rating = np.mean(original_ratings)

            substitution_ratings = substitution.loc[(i,["rating_1","rating_2",
                                                        "rating_3","rating_4"])].values
            substitution_ratings = np.array(substitution_ratings, dtype = float)
            substitution_ratings = substitution_ratings[~np.isnan(substitution_ratings)]
            if not len(substitution_ratings == 2):
                assert item[0] == "illegal" and item[1] == "criminal" and i == 7
                assert len(substitution_ratings) == 1
            substitution_rating = np.mean(substitution_ratings)

            avg_diffs.append(substitution_rating - original_rating)
        avg_diffs = [ 1 if ad > 0 else 0 for ad in avg_diffs ]
        if sum(avg_diffs) >= 13:
            exclude.append(item)

    return exclude

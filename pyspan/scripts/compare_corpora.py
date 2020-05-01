import os
import pickle
from pyspan.bitcounters import BitCounter
from pyspan.config import *
from pyspan import count_words
from scipy import stats

# Correlate distributions
# PKL
pkl_crec = pickle.load(open("{}partial_kls-unigrams".format(crec_paths["metrics_dir"]), "rb"))[["dmetric","rmetric"]]
pkl_debates = pickle.load(open("{}partial_kls-unigrams".format(debate_paths["metrics_dir"]), "rb"))[["dmetric","rmetric"]]
pkl_debates.rename(columns = {"dmetric": "dmetric_debates",
                              "rmetric": "rmetric_debates"}, inplace = True)
pkl_all = pkl_crec.join(pkl_debates, how = "outer")
pkl_all = pkl_all.fillna(0)
print "Correlation between PKL_D: {}".format(stats.pearsonr(pkl_all["dmetric"],
                                                            pkl_all["dmetric_debates"])[0])
print "Correlation between PKL_R: {}".format(stats.pearsonr(pkl_all["rmetric"],
                                                            pkl_all["rmetric_debates"])[0])

# Log p's
logps_crec = pickle.load(open("{}signals-unigrams".format(crec_paths["metrics_dir"]), "rb"))[["dmetric","rmetric"]]
logps_debates = pickle.load(open("{}signals-unigrams".format(debate_paths["metrics_dir"]), "rb"))[["dmetric","rmetric"]]
logps_debates.rename(columns = {"dmetric": "dmetric_debates",
                                "rmetric": "rmetric_debates"}, inplace = True)
logps_all = logps_crec.join(logps_debates, how = "outer")
logps_all = logps_all.fillna(0)
print "Correlation between Log P_D: {}".format(stats.pearsonr(logps_all["dmetric"],
                                                              logps_all["dmetric_debates"])[0])
print "Correlation between Log P_R: {}".format(stats.pearsonr(logps_all["rmetric"],
                                                              logps_all["rmetric_debates"])[0])

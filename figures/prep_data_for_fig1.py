import numpy as np
import pickle
from pyspan.config import *

df = pickle.load(open("{}signals-unigrams".format(paths["metrics_dir"], "rb")))
np.savetxt("rmetric", df.rmetric.values)

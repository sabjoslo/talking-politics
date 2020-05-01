import logging
import numpy as np
import pandas as pd
import pickle
import re
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm
import string
import time
from pyspan.config import *
log_dir = paths["log_dir"]
if not ganesha:
    import matplotlib as mpl
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    # https://stackoverflow.com/questions/19310735#19319972
    def extended(ax, x, y, **args):

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_ext = np.linspace(xlim[0], xlim[1], 100)
        p = np.polyfit(x, y , deg=1)
        y_ext = np.poly1d(p)(x_ext)
        ax.plot(x_ext, y_ext, **args)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    #https://chris35wills.github.io/matplotlib_diverging_colorbar/
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin = None, vmax = None, midpoint = None,
                    clip = False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip = None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, .5, 1]
            return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def startLog(fn):
    filename = "{}{}.log".format(log_dir, fn)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO, filename=filename)

# A custom timestamp to integer function to cope with timestamps outside of
# Unix epochs
def timestamp_to_integer(timestamp, fmt):
    time_obj=time.strptime(timestamp, fmt)
    nyear, nday=time_obj.tm_year, time_obj.tm_yday
    return 365*nyear+nday

def approx_eq(a, b, tol = 1e-8):
    return abs(a-b) <= tol

def tokenize(text, ignore_ascii = False):
    text = re.sub('[%s]'%re.escape(string.punctuation), ' ', text)
    if ignore_ascii:
        return [ word.lower().encode("ascii", "ignore") for word in
                 text.replace("\n", " ").split() ]
    else:
        return [ word.lower() for word in
                 text.replace("\n", " ").split() ]

def get_metric(metrics, token, party):
    assert party in ("democrat", "republican")
    col = "dmetric" if party == "democrat" else "rmetric"
    if token not in metrics.index:
        return np.nan
    return metrics.loc[token][col]

def get_partisan_score(excerpt, metrics, f = np.prod):
    tokens = tokenize(excerpt)

    dmetrics = np.array(map(lambda token: get_metric(metrics, token, "democrat"),
                            tokens))
    dmetrics = dmetrics[~np.isnan(dmetrics)]
    rmetrics = np.array(map(lambda token: get_metric(metrics, token, "republican"),
                            tokens))
    rmetrics = rmetrics[~np.isnan(rmetrics)]

    # Calculate metric for the entire excerpt
    dscore = f(dmetrics)
    rscore = f(rmetrics)

    return dscore, rscore

# Replace missing data with its mean
def replace_nans(arr):
    arr[np.isnan(arr)] = np.mean(arr[~np.isnan(arr)])
    return arr

# Return logistic CDF(x)
def sigmoid(x):
    return np.exp(x)/(1 + np.exp(x))

# Return a statsmodels.OLS object fitted to the data passed as argument
def linear_regression(Y, X, add_constant = True):
    Y, X = np.array(Y), np.array(X)
    if add_constant:
        X = sm.add_constant(X)
    mod = sm.OLS(Y, X)
    return mod.fit()

# Return an sklearn.linear_model.LogisticRegression object fitted to the data
# passed as argument. Optionally, set `lasso` to True to introduce an L1 penalty
# (enforce sparsity in the vector of coefficients).
def logistic_regression(Y, X, lasso = False, **kwargs):
    if lasso:
        kwargs["penalty"] = kwargs.get("penalty", "l1")
        kwargs["solver"] = kwargs.get("solver", "liblinear")
        assert kwargs["penalty"] == "l1"
        assert kwargs["solver"] == "liblinear"

    logit = linear_model.LogisticRegressionCV(**kwargs)
    logit.fit(X, Y)
    return logit

# Takes a matrix X as input. If shape_ == "long", each row of X is assumed to
# be an observation, and each column is assumed to correspond to a variable. If
# shape_ == "wide", the reverse should be true (shape_ defaults to "long").
#
# Plots the correlation matrix of X. If a filename is specified, saves to that
# filename.
def plot_corrs(X, shape_ = "long", labels = None, filename = None,
               cmap = "seismic"):
    assert shape_ in ("long", "wide")
    if shape_ == "wide":
        X = X.T
    n = X.shape[1]
    if not labels:
        labels = [""] * n
    assert len(labels) == n

    corrs = [ stats.pearsonr(X[:,i], X[:,j])[0] for i in range(n) for j in
              range(n) ]
    corrs = np.array(corrs).reshape((n, n))
    for i in range(n):
        corrs[i,i] = np.nan

    fig = plt.figure()
    plt.imshow(corrs, cmap = cmap, clim = (-1, 1),
               norm = MidpointNormalize(midpoint = 0, vmin = -1, vmax = 1))
    plt.xticks(range(n), labels, rotation = 90, fontsize = 20)
    plt.yticks(range(n), labels, fontsize = 20)
    plt.colorbar()
    plt.title("Pairwise correlations", fontsize = 20)
    if filename:
        plt.savefig(filename,
                    bbox_inches = mpl.transforms.Bbox([[-1, -1.25], [6, 4]]))
    return fig

# Plot the histogram of the X argument, along with lines indicating the mean
# and standard deviation. If a filename is specified, saves to that filename.
def histogram(X, bins = 50, title = "Histogram", filename = None, **kwargs):
    mu, se = np.mean(X), stats.sem(X)

    fig = plt.figure(**kwargs)
    plt.hist(X, bins = bins)
    ylim = plt.ylim()
    plt.ylim(*ylim)
    # Indicate the mean
    plt.plot([ mu, mu ], [ 0, ylim[1] ])
    # Indicate one SE above and below the mean
    plt.plot([ mu-se, mu-se ], [ 0, ylim[1] ], color = "orange",
             linestyle = "--")
    plt.plot([ mu+se, mu+se ], [ 0, ylim[1] ], color = "orange",
             linestyle = "--")
    plt.title(title)
    if filename:
        plt.savefig(filename)
    return fig, mu, se

# Turn a list of lists into a csv
def to_csv(X, filename, **kwargs):
    if filename[-4:] != ".csv":
        filename += ".csv"
    df = pd.DataFrame(X, **kwargs)
    df.to_csv(filename)

# Generate a random sequence of four letters to use as an identifier when
# making bootstrapped samples
def make_id():
    return "".join(np.random.choice(list(string.lowercase), 4))

def is_nan(x):
    if not isinstance(x, np.float):
        return False
    return np.isnan(x)
is_nan = np.vectorize(is_nan)

def plot_prf(gam, term, coef = None):
    XX = gam.generate_X_grid(term = term)
    pdeps, conf_intervals = gam.partial_dependence(term = term, X = XX,
                                                   width = .95)
    order_ixs = np.argsort(XX[:,term])
    x = XX[order_ixs,term]
    pdeps = pdeps[order_ixs]
    conf_intervals = conf_intervals[order_ixs,:]
    plt.plot(x, pdeps)
    plt.fill_between(x, conf_intervals[:,0], conf_intervals[:,1],
                     alpha = .5)
    plt.plot(x, coef * x)

# Return a correlation between vectors X and Y where elements with missing
# values in X or Y are removed
def pearsonr_(x, y):
    return stats.pearsonr(x[(~is_nan(x)) & (~is_nan(y))],
                          y[(~is_nan(x)) & (~is_nan(y))])

# https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
def cohensd(a1, a2):
    diff = np.mean(a1) - np.mean(a2)
    s1 = np.std(a1, ddof = 1)
    s2 = np.std(a2, ddof = 1)
    n1 = len(a1)
    n2 = len(a2)
    spooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return diff / spooled

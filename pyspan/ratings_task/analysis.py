from __future__ import division

from copy import deepcopy
from scipy import stats
from sklearn.metrics import roc_curve, auc
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from pyspan.ratings_task.data import *

class PerceptualData(object):
    # If binary is True, bin responses into one of two bins: "Democratic" or
    # "Republican". The default is binary == False, because it doesn't make much
    # difference.
    def __init__(self, binary = False, ixs = p_ixs + a_ixs, **kwargs):
        self.binary = binary
        self.ixs = ixs
        self.words = all_.loc[self.ixs]["word"].values
        self.data = minidf
        for k, v in kwargs.items():
            self.data = self.data.loc[self.data[k] == v]

    def get_discriminability_by_word(self):
        self.discrim_by_word = []
        for ix in self.ixs:
            col = self.data[str(ix)]
            col = col[~np.isnan(col)]
            perceptions = col
            frac_r = np.mean(map(lambda w: w >= 3, col))
            self.discrim_by_word.append((np.mean(perceptions),
                                         stats.sem(perceptions), frac_r))
        self.discrim_by_word = np.array(self.discrim_by_word)

    def word_map(self, func = get_pkls, filename = None, **kwargs):
        fig = plt.figure(**kwargs)
        s_w = sorted(enumerate(self.discrim_by_word[:,0]), key = lambda p: p[1])
        x__ = list(itertools.chain(*[ [func(self.ixs[w])]*11 for w, x___ in s_w
                                    ]))
        # Pad if there aren't enough observations
        if len(x__)%22 != 0:
            x__ += [0]*11

        x__ = np.array(x__).reshape(int(len(x__)/22), 22)
        plt.imshow(x__, cmap="seismic", clim = (np.min(x__), np.max(x__)),
                   norm = MidpointNormalize(midpoint = 0, vmin = np.min(x__),
                   vmax = np.max(x__)), alpha = .5)
        plt.xticks([])
        plt.yticks([])
        breakpt = sorted(list(self.discrim_by_word[:,0]) + [2.5]).index(2.5)
        y = int(breakpt / 2) - .5
        if breakpt % 2 == 0:
            plt.plot([ -.5, 21.5 ], [ y, y ], linewidth = 6, color = "m")
        else:
            plt.plot([ -.5, 10.5 ], [ y+1, y+1 ], linewidth = 6, color = "m")
            plt.plot([ 10.5, 10.5 ], [ y, y+1 ], linewidth = 6, color = "m")
            plt.plot([ 10.5, 21.5 ], [ y, y ], linewidth = 6, color = "m")

        plt.axis("off")
        for i in range(len(s_w)):
            ix, w = s_w[i]
            plt.text(5 + 11*(i%2==1), int(i/2), all_.loc[self.ixs[ix]]["word"],
                     ha="center", va="center", color="k")

        if filename:
            plt.savefig(filename)

    def correlational_analysis(self, df, adjust_x_axis = True, midpoint = 0,
                               savetofile = False, sepl = 0,
                               standardize_x = False, xlabel = "Metric",
                               verbose = True):
        ixs = np.array(range(len(self.ixs)))

        xd = np.array([ df.loc[w]["dmetric"] for w in self.words ])
        xr = np.array([ df.loc[w]["rmetric"] for w in self.words ])
        assert np.array_equal(ixs[~np.isnan(xd)], ixs[~np.isnan(xr)])
        ixs = ixs[~np.isnan(xd)]
        xd = xd[ixs]
        xr = xr[ixs]

        y = self.discrim_by_word[:,0]
        y = y[ixs]
        yerr = self.discrim_by_word[:,1]
        yerr = yerr[ixs]
        frac_r = self.discrim_by_word[:,2]
        frac_r = frac_r[ixs]

        dat = zip(self.words, xd, xr, y, yerr, frac_r)

        if savetofile:
            to_csv(dat, filename = savetofile, columns = [ "word", "dmetric",
                   "rmetric", "ratings_mean", "ratings_std", "frac_r" ])

        if standardize_x:
            mu, std = np.mean(xr), np.std(xr)
            x = (xr - mu)/std
            sepl = (sepl - mu)/std
        else:
            x = xr

        fig, ax = plt.subplots(1, figsize = (40, 32))
        if adjust_x_axis:
            x_rad = max(max(x), -min(x))
            x_rad *= 1.2
            ax.set_xlim(-x_rad, x_rad)
        seismic = plt.cm.ScalarMappable(norm = MidpointNormalize(midpoint = midpoint,
                                                                 vmin = min(x),
                                                                 vmax = max(x)),
                                        cmap = "seismic")
        color = seismic.to_rgba(x)
        ax.scatter(x, y, s = 200, color = color)
        ax.errorbar(x, y, yerr = yerr, fmt = ".", ms = 0, color = color)
        ax.set_xlim(*ax.get_xlim())
        # Set y axis to cover the full range of possible ratings
        ax.set_ylim(0,5)
        ax.plot([ sepl, sepl ], [ plt.ylim()[0], plt.ylim()[1] ], "k")
        ax.plot([ plt.xlim()[0], plt.xlim()[1] ], [ 2.5, 2.5 ], "k")
        # Add linear trend
        X = sm.add_constant(x)
        mod = sm.OLS(y, X)
        res = mod.fit()
        m0, m1 = res.params
        s0, s1 = res.bse

        if verbose:
            print res.summary()
            print "\n\n==========\n\n"
            print "Pearson's R: %f (p-val: %f)"%stats.pearsonr(x, y)
            print "Spearman's rho: %f (p-val: %f)"%stats.spearmanr(x, y)

        x_edges = [ plt.xlim()[0], plt.xlim()[1] ]
        y_edges = res.predict(sm.add_constant(x_edges))
        ax.plot(x, res.fittedvalues, x_edges, y_edges, color="purple",
                linewidth = 2)
        prstd, iv_l, iv_u = wls_prediction_std(res)
        ax = extended(ax, x, iv_l, linestyle = "--", color = "purple",
                      linewidth = 2)
        ax = extended(ax, x, iv_u, linestyle = "--", color = "purple",
                      linewidth = 2)
        ax.set_xlabel(xlabel, fontsize = 50)
        ax.xaxis.set_tick_params(labelsize = 30)
        ax.set_ylabel("Avg. rating", fontsize = 50)
        ax.yaxis.set_tick_params(labelsize = 30)
        # Set tick labels that are the rating+1, for interpretability
        ax.set_yticks(range(6))
        ax.set_yticklabels(range(1,7))
        for i, w in enumerate(self.words[ixs]):
            ax.annotate(w, xy = (x[i], y[i]), fontsize = 30)

        if savetofile:
            plt.savefig(savetofile)

class SparseLR(object):
    def __init__(self, Y, X, var_labs = []):
        assert len(Y) == X.shape[0]
        assert len(var_labs) in (0, X.shape[1])
        self.Y = Y
        self.X = X
        self.n = len(self.Y)
        Y_ = list(Y)
        self.base_rate = Y_.count(1)/self.n
        self.var_labs = var_labs

        self.model = logistic_regression(Y, X, lasso = True)
        self.coef = self.model.coef_[0]
        # Get score
        self.score = self.model.score(self.X, self.Y)
        # Get predictions and loss
        self.preds = self.model.predict_proba(self.X)
        self.loss = [ p[0] if y else p[1] for p, y in zip(self.preds, self.Y) ]
        self.log_loss = [ math.log(l) for l in self.loss ]
        # ROC AUC
        fpr, tpr, thresholds = roc_curve(self.Y, self.preds[:,1])
        self.roc = np.array(zip(fpr, tpr, thresholds))
        self.auc = auc(fpr, tpr)

    # Print marginal effects (e^\beta)
    def print_marg_effects(self):
        marg_effects = map(math.exp, self.coef)
        if len(self.var_labs) > 0:
            marg_effects = zip(self.var_labs, marg_effects)
        pp.pprint(marg_effects)

    # Create a histogram of the loss
    def loss_histogram(self, bins = 100, filename = None, **kwargs):
        fig = plt.figure(**kwargs)
        plt.hist(self.loss, bins = bins)
        plt.title("Histogram of loss")
        if filename:
            plt.savefig(filename)
        fig.show()

    # Takes a variable (a column of x or a variable label) as input, and plots
    # the loss as a function of the value of that variable. If `log` == True
    # (default), plots the log loss. If `filename` is passed, saves a copy of
    # the plot to `filename`.
    def plot_loss(self, var, log = True, filename = None, **kwargs):
        assert isinstance(var, int) or isinstance(var, basestring)
        if isinstance(var, basestring):
            var = self.var_labs.index(var)
        xlab = "X[:,{}]".format(var)
        if len(self.var_labs) > 0:
            xlab = self.var_labs[var]
        y, ylab = self.loss, "Loss"
        if log:
            y, lab = self.log_loss, "Log loss"

        fig = plt.figure(**kwargs)
        plt.plot(self.X[:,var], y, ".")
        xlim = plt.xlim()
        plt.xlim(*xlim)
        plt.plot(xlim, [ math.log(.5), math.log(.5) ], "--")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if filename:
            plt.savefig(filename)
        plt.show()

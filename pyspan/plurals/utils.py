from __future__ import division
import numpy as np

def calc_proportions(arr, l1, l2):
    n1 = np.count_nonzero(np.in1d(arr, l1))
    n2 = np.count_nonzero(np.in1d(arr, l2))
    assert n1 + n2 <= len(l1)

    return n1 / ( n1 + n2 )

def proportions_as_dataframe(df, cols, *args):
    summary = df[cols]
    dat = df[range(100, 200)].values
    props = np.apply_along_axis(lambda a: calc_proportions(a, *args), 1, dat)
    summary["p"] = props
    return summary

import itertools
from scipy import stats
from pyspan.config import *
METRICS_DIR = paths["metrics_dir"]

def get_vectors(fn):
    dy=[]
    ry=[]
    with open(fn, 'r') as fh:
        line=fh.readline()
        line=fh.readline()
        while line.strip():
            phrase,dmetric,rmetric=line.strip().split('|')
            dmetric,rmetric=float(dmetric),float(rmetric)
            dy.append(dmetric)
            ry.append(rmetric)
            line=fh.readline()
    return dy,ry

def get_pearson_r(x,y):
    return stats.pearsonr(x,y)

def get_dict_of_vecs(metrics=None, mode='bigrams'):
    vecs=dict()
    if isinstance(metrics,type(None)):
        metrics=[ 'partial_kls','probabilities','signals' ]
    metrics=[ '%s-%s'%(metric, mode) for metric in metrics ]
    for metric in metrics:
        dy,ry=get_vectors(METRICS_DIR+metric)
        vecs['{} (D)'.format(metric)]=dy
        vecs['{} (R)'.format(metric)]=ry
    return vecs

def get_correlations(metrics=None, mode='bigrams', fn=None):
    with open(METRICS_DIR+'correlations-{}'.format(mode),'w') as fh:
        fh.write(' | |Pearson CC|p-value')

        vecs=get_dict_of_vecs(metrics=metrics, mode=mode)
        for v0,v1 in itertools.combinations(vecs.keys(),2):
            cc,pval=get_pearson_r(vecs[v0],vecs[v1])
            fh.write('\n{}|{}|{}|{}'.format(v0,v1,cc,pval))

def sort_metrics(metric='partial_kls', mode='bigrams'):
    vecs=get_dict_of_vecs(metrics=[metric], mode=mode)

    for party in ('D', 'R'):
        diffs=sorted(enumerate(vecs['%s-%s (%s)'%(metric, mode, party)]),
                     key=lambda x:x[1])
        with open(METRICS_DIR+'%s-%s'%(metric, mode), 'r') as rfh:
            # Get rid of header
            rfh.readline()
            data=rfh.read().split('\n')
        with open(METRICS_DIR+'sorted_%s_%s-%s'%(metric, mode, party),
                  'w') as wfh:
            for diff in reversed(diffs):
                wfh.write(data[diff[0]]+'\n')

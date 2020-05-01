from collections import defaultdict
from itertools import chain
from pyspan.config import *
years_=map(str, settings["years"]) if settings["mode"] == "crec" else map(str, settings["election_cycles"])
PROC_TXT_DIR = paths["proc_txt_dir"]
NGRAM_DIR = paths["ngram_dir"]
import os
import re
import string
from pyspan.Purger import Purger
purger = Purger(years = range(2012, 2018), overwrite = True)

def process(text_):
    tokens = text_.split()
    tokens = filter(lambda tok:not tok.isdigit() and tok.strip(), tokens)
    tokens = [ tok for tok in tokens if tok not in purger.stopwords ]
    return tokens

def prune_ngram_counts(dgrams, rgrams, tol = 10):
    for d,d_ in ((dgrams, rgrams), (rgrams, dgrams)):
        for k in d.keys():
            if d[k]<tol:
                if k not in d_:
                    del d[k]
                elif d_[k]<tol:
                    del d[k]
                    del d_[k]

    return dgrams, rgrams

def count_unigrams():
    for year in os.listdir(PROC_TXT_DIR):
        if year not in years_:
            continue
        dugrams=defaultdict(lambda:0)
        rugrams=defaultdict(lambda:0)
        ydir='{}{}/'.format(PROC_TXT_DIR, year)
        with open(NGRAM_DIR+'unigrams-{}.txt'.format(year), 'w') as wfh:
            wfh.write('party|phrase|count\n')
            files = list(chain(*[ [ "{}/{}".format(dir_[0], f) for f in dir_[2]
                                    if f != ".DS_Store" ] for dir_ in
                                    os.walk(ydir) ]))
            dfiles = [ f for f in files if f.split("/")[-1] == "dem_speech" ]
            rfiles = [ f for f in files if f.split("/")[-1] == "repub_speech" ]
            for f in dfiles:
                with open(f, 'r') as rfh:
                        tokens=process(rfh.read())
                        for ugram in tokens:
                            dugrams[ugram]+=1
            for f in rfiles:
                with open(f, 'r') as rfh:
                        tokens=process(rfh.read())
                        for ugram in tokens:
                            rugrams[ugram]+=1

            dugrams, rugrams = prune_ngram_counts(dugrams, rugrams)

            wfh.write('\n'.join([ 'D|{}|{}'.format(k,dugrams[k]) for k in
                                  dugrams.keys() ]))
            wfh.write('\n')
            wfh.write('\n'.join([ 'R|{}|{}'.format(k,rugrams[k]) for k in
                                  rugrams.keys() ]))

def count_bigrams():
     for year in os.listdir(PROC_TXT_DIR):
        if year not in years_:
            continue
        dbigrams=defaultdict(lambda:0)
        rbigrams=defaultdict(lambda:0)
        ydir='{}{}/'.format(PROC_TXT_DIR, year)
        with open(NGRAM_DIR+'bigrams-{}.txt'.format(year), 'w') as wfh:
            wfh.write('party|phrase|count\n')
            files = list(chain(*[ [ "{}/{}".format(dir_[0], f) for f in dir_[2]
                                    if f != ".DS_Store" ] for dir_ in
                                    os.walk(ydir) ]))
            dfiles = [ f for f in files if f.split("/")[-1] == "dem_speech" ]
            rfiles = [ f for f in files if f.split("/")[-1] == "repub_speech" ]
            for f in dfiles:
                with open(f, 'r') as rfh:
                        text=rfh.read()
                        for sentence in text.split('\n'):
                            tokens=process(sentence)
                            for i in range(len(tokens)-1):
                                bigram=' '.join(tokens[i:i+2])
                                dbigrams[bigram]+=1
            for f in rfiles:
                with open(f, 'r') as rfh:
                        text=rfh.read()
                        for sentence in text.split('\n'):
                            tokens=process(sentence)
                            for i in range(len(tokens)-1):
                                bigram=' '.join(tokens[i:i+2])
                                rbigrams[bigram]+=1

            dbigrams, rbigrams = prune_ngram_counts(dbigrams, rbigrams)

            wfh.write('\n'.join([ 'D|{}|{}'.format(k,dbigrams[k]) for k in
                                  dbigrams.keys() ]))
            wfh.write('\n')
            wfh.write('\n'.join([ 'R|{}|{}'.format(k,rbigrams[k]) for k in
                                  rbigrams.keys() ]))

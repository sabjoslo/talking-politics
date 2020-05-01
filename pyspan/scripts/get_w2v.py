#!/usr/bin/python2.7
"""1. Concatenate all plain text data files and pipe it all into a single file named as `corpus_fn`.
2. `./get_w2v.py` will train and save to disk a word2vec model on the data in this corpus.
"""

import os
from pyspan.config import paths
from wordplay import core

corpus_fn = "/data/sabina/crec_corpus"

td=core.token_distributions(corpus_fn=corpus_fn, lemmatize=False)
td.word2vec_("{}crec_w2v".format(paths["input_dir"]))

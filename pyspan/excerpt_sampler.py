"""Extract sentences from the Congressional Record.
"""

import numpy as np
import pandas as pd
import os
import pickle
import re
import signal
import sys
import time
from wordplay.utils import *
from pyspan.config import *
PROC_TXT_DIR = paths["proc_txt_dir"]
OUTPUT_DIR = paths["output_dir"]
log_dir = paths["log_dir"]
years = settings["years"]

class Sampler(object):
    def __init__(self, words):
        self.words = words
        if " " in words[0]:
            assert "y" in raw_input("Bigrams detected. Is this the correct mode?").lower()
            self.mode = "bigrams"
        else:
            assert "y" in raw_input("Unigrams detected. Is this the correct mode?").lower()
            self.mode = "unigrams"

    def build_corpus(self, years = years, party = "both", save = True):
        if isinstance(years, type(None)):
            years = os.listdir(PROC_TXT_DIR)
        assert hasattr(years, "__iter__")
        assert party in ("democrat", "republican", "both")
        if party == "democrat":
            fs = [ "dem_sentences" ]
        elif party == "republican":
            fs = [ "repub_sentences" ]
        else:
            fs = [ "dem_sentences", "repub_sentences" ]
        corpus = []
        for year in years:
            for dir_, _, files in os.walk(PROC_TXT_DIR + str(year)):
                files = [ f for f in files if f in fs ]
                for f in files:
                    fn = "{}/{}".format(dir_, f)
                    df = pickle.load(open(fn, "rb"))
                    excerpts = df["speech"].values
                    corpus.extend(map(to_ascii,
                                      filter(lambda s: s.strip(),
                                             excerpts)))
        self.corpus = corpus
        if save:
            fn = "{}sentences_{}_{}".format(OUTPUT_DIR, party,
                                         "-".join(map(str, years)))
            with open(fn, "w") as wfh:
                wfh.write("\n".join(corpus))

    def load_corpus(self, years = years, party = "both"):
        if isinstance(years, type(None)):
            years = os.listdir(PROC_TXT_DIR)
        assert hasattr(years, "__iter__")
        assert party in ("democrat", "republican", "both")
        fn = "{}sentences_{}_{}".format(OUTPUT_DIR, party,
                                     "-".join(map(str, years)))
        self.corpus = open(fn, "r").read().split("\n")

    def sample(self, word, n):
        relevant = filter(lambda s: re.search(r"(?<!\w){}(?!\w)".format(word),
                                              s), self.corpus)
        return np.random.choice(relevant, n)

    def get_excerpts(self, n = 3, save = True, fn = None):
        self.excerpts = dict([ (word, self.sample(word, n)) for word in
                               self.words ])
        if save:
            if isinstance(fn, type(None)):
                fn = OUTPUT_DIR + "excerpts-" + self.mode
            with open(fn, "wb") as wfh:
                pickle.dump(self.excerpts, wfh)

    def categorize_words(self, df = None, fn = None, document = True,
                         logfile = "categorization_output"):
        # If I encounter a SIGINT, first do some cleanup. This will enable me
        # to stop and start categorization.
        def sigint_handler(signal_, frame):
            print "\nKeyboard interrupt. Cleaning up and saving to dataframe."
            clean_up()
            sys.exit(0)

        def clean_up():
            df_ = pd.DataFrame(d.items(), columns=[ "word", "class" ])
            df_ = df_.set_index("word")
            df_ = df_.reindex(self.words[:len(df_)])
            with open(OUTPUT_DIR + fn, "wb") as wfh:
                pickle.dump(df_, wfh)
            print "Saved to {}.".format(fn)
            if document:
                lfh.write("\n=====\n")
                lfh.close()

        assert hasattr(self, "excerpts")

        assert isinstance(df, type(None)) or isinstance(fn, type(None))
        if not isinstance(fn, type(None)):
            df = pickle.load(open(OUTPUT_DIR + fn, "rb"))
        else:
            fn = "categorized-" + self.mode
            if isinstance(df, type(None)):
                df = pd.DataFrame()
        assert list(df.index) == self.words[:len(df.index)]
        if document:
            lfh = open("{}{}-{}".format(log_dir, logfile, self.mode), "a")
            lfh.write(time.strftime("%c"))

        words_ = self.words[len(df.index):]
        if not df.empty:
            d = dict(zip(df.index, df["class"]))
        else:
            d = dict()
        signal.signal(signal.SIGINT, sigint_handler)
        for word in words_:
            excerpts = self.excerpts[word]
            excerpts = "\n\n".join(excerpts)
            excerpts = excerpts.replace(" {} ".format(word), " \033[41;37m{}\033[m ".format(word))
            excerpts = "\n=====\n" + word + ":\n\n" + excerpts
            val = raw_input(excerpts + "\n")
            if document:
                lfh.write("""
=====

{}

-----

{}

-----

{}

=====

""".format(word, excerpts, val))
            d[word] = val
        clean_up()

    # The following method converts excerpts and word lists to a format more
    # amenable to RA coding.
    def excerpts_to_tex(self, tex_fn = "excerpts", csv_fn = "words"):
          with open("{}{}-{}.tex".format(OUTPUT_DIR, tex_fn, self.mode), "w") as texfh:
            texfh.write("""\\documentclass{article}
\\usepackage{color}

\\begin{document}""")
            with open("{}{}-{}.csv".format(OUTPUT_DIR, csv_fn, self.mode), "w") as csvfh:
                csvfh.write("word,class\n")
                for k, v in self.excerpts.iteritems():
                    texfh.write("\n\\section*{{{}}}\n".format(k) + "\n\\vspace{8mm}\n".join([
                               re.sub(r"(?<!\w){}(?!\w)".format(k),
                                      "{{\\\\bf \\color{{red}} {}}}".format(k),
                                      v_) for v_ in v ]) + "\\pagebreak\n"
                              )
                    csvfh.write("{},\n".format(k))
                texfh.write("\n\\end{document}")

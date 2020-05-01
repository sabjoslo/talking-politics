#!/usr/bin/python2

import os
import pickle
from pyspan.config import *
years = settings["years"]
OUTPUT_DIR = paths["output_dir"]
from pyspan.excerpt_sampler import Sampler

fn = os.path.expanduser("survey_terms1.txt")
words = filter(lambda s: s.strip(), open(fn, "r").read().split("\n"))
sampler = Sampler(words = words)

excerpts_fn = "excerpts"

def prep():
    sampler.build_corpus(years = years)
    sampler.get_excerpts()

def categorize():
    sampler.excerpts = pickle.load(open(excerpts_fn, "rb"))
    sampler.categorize_words()

def restart():
    sampler.excerpts = pickle.load(open(excerpts_fn, "rb"))
    sampler.categorize_words(fn = "categorized")

if __name__ == "__main__":
    prep()

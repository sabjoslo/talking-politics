from itertools import chain
from pyspan.excerpt_sampler import Sampler
from pyspan.procedural_words import ProceduralWords

if __name__ == "__main__":
    # Get words
    pw = ProceduralWords(mode = "bigrams")
    pw.load_df()
    bigrams = pw.filter_()
    unigrams = list(chain(*[ pw_.split() for pw_ in bigrams ]))

    for grams in (bigrams,):
        # Make materials
        sampler = Sampler(grams)
        sampler.load_corpus()
        sampler.get_excerpts()
        sampler.excerpts_to_tex()

from google.cloud import language
client = language.LanguageServiceClient()
from google.cloud.language import enums
from google.cloud.language import types
import numpy as np
import pandas as pd
from pattern.en import sentiment
from pyspan.config import *
INPUT_DIR = paths["input_dir"]

# The Warriner et al. data accompanies the online version of the paper at
# http://dx.doi.org/10.3758/s13428-012-0314-x
CRR_DF = pd.read_csv(INPUT_DIR + "Ratings_Warriner_et_al.csv")
CRR_DF = CRR_DF[["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]]
CRR_DF.rename(columns = {"Word": "word"}, inplace = True)
CRR_DF = CRR_DF.set_index("word")

# The CRR valence data only contains word lemmas (i.e. no plurals or
# conjugations). This is a running dict of non-lemmatized words that we've
# used in studies and their corresponding lemmas, so if we don't have CRR data
# on the word we want, we can look up its lemma.
LEMMA_DICT = { # LoP_Ratings_2
               "women": "woman", "republicans": "republican",
               "cuts": "cut", "americans": "american",
               "families": "family", "communities": "community",
               "benefits": "benefit", "millions": "million",
               "students": "student", "protections": "protection",
               "going": "go", "regulations": "regulation", "got": "have",
               "spending": "spend", "years": "year", "things": "thing",
               "folks": "folk", "businesses": "business", "days": "day",
               "states": "state", "bureaucrats": "bureaucrat",
               "plentiful": "plenty", "qualified": "qualify",
               # LoP_2AFC_2
               "months": "month", "indicated": "indicate", "stated": "state",
               "immigrants": "immigrant", "aliens": "alien", "wanted": "want",
               "seconds": "second", "minutes": "minute", "companies": "company",
               "manufacturers": "manufacturer", "changes": "change",
               "reforms": "reform", "values": "value",
               "underserved": "underserve", "problems": "problem",
               "adopted": "adopt", "passed": "pass", "cities": "city",
               "counties": "county", "eliminates": "eliminate",
               "provides": "provide", "taxes": "tax", "workers": "worker",
               "employees": "employee", "parks": "park", "forests": "forest",
               "lives": "life", "talking": "talk", "colleagues": "colleague",
               "friends": "friend", "seniors": "senior", "patients": "patient",
               "brought": "bring", "bringing": "bring", "supported": "support",
               "sponsored": "sponsor", "ensures": "ensure",
               "protecting": "protect", "defending": "defend", "rates": "rate",
               "prices": "price", "consumers": "consumer",
               "taxpayers": "taxpayer", "babies": "baby", "policies": "policy",
               "leaving": "leave", "left": "leave", "comments": "comment",
               "homeowners": "homeowner", "farmers": "farmer",
               "investors": "investor", "inventors": "inventor",
               "commitments": "commitment", "promises": "promise", "wars": "war",
               "questions": "question", "programs": "program",
               "activities": "activity", "ports": "port", "borders": "border",
               "partners": "partner", "allies": "ally",
               "corporations": "corporation", "hoosiers": "hoosier",
               "taking": "take", "takes": "take", "increases": "increase" }

def _get_crr_scores(w, treat_na = "strict"):
    try:
        return CRR_DF.loc[w][["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]].values
    except KeyError:
        try:
            return CRR_DF.loc[LEMMA_DICT[w]][["V.Mean.Sum", "A.Mean.Sum",
                                           "D.Mean.Sum"]].values
        except KeyError:
            if treat_na == "strict":
                return [np.nan, np.nan, np.nan]
            return [5, 5, 5] # 5 corresponds to neutral

def get_valence(excerpt, use = "crr", treat_na = "strict"):
    # The Google Natural Language API returns a score and magnitude. From the
    # docs
    # (https://cloud.google.com/natural-language/docs/basics#sentiment-analysis-values):
    #
    # score of the sentiment ranges between -1.0 (negative) and 1.0 (positive)
    #   and corresponds to the overall emotional leaning of the text.
    # magnitude indicates the overall strength of emotion (both positive and
    #   negative) within the given text, between 0.0 and +inf. Unlike score,
    #   magnitude is not normalized; each expression of emotion within the text
    #   (both positive and negative) contributes to the text's magnitude (so
    #   longer text blocks may have greater magnitudes).
    if use == "google":
        document = types.Document(content = excerpt,
                                  type = enums.Document.Type.PLAIN_TEXT)
        annotations = client.analyze_sentiment(document = document)
        return ( annotations.document_sentiment.score,
                 annotations.document_sentiment.magnitude )

    # The pattern API returns a polarity and subjectivity score. From the docs
    # (https://www.clips.uantwerpen.be/pages/pattern-en#sentiment):
    #
    # The sentiment() function returns a (polarity, subjectivity)-tuple for the
    # given sentence, based on the adjectives it contains, where polarity is a
    # value between -1.0 and +1.0 and subjectivity between 0.0 and 1.0.
    elif use == "pattern":
        return sentiment(excerpt)

    # The Center for Reading Research data contains scores for valence, arousal
    # and dominance. From Warriner et al. (2013):
    #
    # **[V]alence** (or pleasantness) of the emotions invoked by a word, going
    # from *unhappy* to *happy*
    #
    # [T]he degree of **arousal** evoked by a word
    #
    # [T]he **dominance**/power of the word--the extent to which the word
    # denotes something that is weak/submissive or strong/dominant
    elif use == "crr":
        assert len(excerpt.split()) == 1 # Not sure yet how to handle n>1-grams
        return _get_crr_scores(excerpt, treat_na = treat_na)

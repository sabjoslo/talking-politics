Materials for stimulus generation and analysis of study 2 reported in "Two kinds of discussion and types of conversation".

# Stimulus generation
The script `pyspan/scripts/get_interchangeable_terms.py` calculates the cosine similarity between all pairs of words in which one word is one of the 5\% most Democratic words, and the other is one of the 5\% most Republican words (on the basis of *PKL*). The cosine similarities between these word pairs is saved as *version_used_for_LoP_2AFC_2_cos_sim_partial_kls-unigrams*. After excluding pairs with proper nouns or abbreviations that were difficult to interpret out of context, we selected the 88 word pairs with the highest cosine similarity. `get_terms.ipynb` generates .txt files of the survey items that [can be uploaded into Qualtrics](https://www.siue.edu/its/qualtrics/pdf/advanced_survey/QualtricsImportExportSurveysAdv.pdf).

# Preprocessing the data
The original data downloaded from Qualtrics is in the csv file `LoP_2AFC_2.csv` (responses to the level of education question have been recoded to be standardized and response ID columns have been removed). The cleaned version, processed by `preprocess.ipynb`, is in the csv file `responses.csv`. Stimuli are indexed by column numbers. The corresponding stimuli are given in `partisan_words.csv` (1--88) and `antonyms.csv` (89--98). 

# Replicating the analysis
The results reported in the paper can be replicated by running the `replicate.ipynb` notebook.

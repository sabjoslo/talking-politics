Materials for stimulus generation and analysis of study 1 reported in "Two kinds of discussion and types of conversation".

# Excluding procedural words
The most recent list of procedural words was not yet compiled when study 1 was run (see `pyspan.Purger.Purger().get_congressional_stopwords()` for code that compiles the most current list of procedural words). For study 1, we took a list of the highest-PKL terms, and excluded post-hoc a subset of the words. These words had been manually determined by one of the authors to be procedural, according to their use in randomly-selected excerpts from the CongRec (these excerpts are saved as a [pickled](https://docs.python.org/2/library/pickle.html) Python dictionary named `excerpts`). The results of this categorization are saved as a pickled [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) named `categorized`. Helper code for the categorization process is included in the Python script `categorize_words_as_procedural.py`.

# Stimulus generation
`get_terms.ipynb` generates .txt files of the survey items that [can be uploaded into Qualtrics](https://www.siue.edu/its/qualtrics/pdf/advanced_survey/QualtricsImportExportSurveysAdv.pdf).

# Preprocessing the data
The original data downloaded from Qualtrics is in the csv file `LoP_Ratings_2.csv`, modified only to exclude response ID columns. The cleaned version, processed by `preprocess.ipynb`, is in the csv file `responses.csv`. Stimuli are indexed by column numbers. The corresponding stimuli are given in `partisan.csv` (1--78) and `antonyms.csv` (79--98).

# Replicating the analysis
The results reported in the paper can be replicated by running the `replicate.ipynb` notebook.

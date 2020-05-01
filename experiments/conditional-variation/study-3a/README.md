 Tje Materials for stimulus generation and analysis of study 3a reported in "Two kinds of discussion and types of conversation".

# Stimulus generation
`get_terms.ipynb` contains the infrastructure to select stimulus items by:
1. Identifying all the words whose polarity cross-validates from the CRec to the debates corpus (saved as `words_dem.csv` and `words_repub.csv`),
2. Reading from a list of these words, manually annotated with their synonyms (saved as `words_dem-annotated.csv` and `words_repub-annotated`), and identify the synonym pairs in which one word is Democratic and the other is Republican, and
3. Allowing manual exclusion of words with multiple meanings (this list is in the file `exclude`).
The list of items is saved as `synonyms.csv`.

This notebook also contains the infrastructure to implement the item-level exclusion criteria, pre-registered [here](https://osf.io/29yac).

`format_for_qualtrics.ipynb` generates .txt files of the survey items that [can be uploaded into Qualtrics](https://www.siue.edu/its/qualtrics/pdf/advanced_survey/QualtricsImportExportSurveysAdv.pdf).

# Preprocessing the data
The original data downloaded from Qualtrics is in the csv file `LoP_Synonyms.csv`, modified only to remove response ID columns. The cleaned version, processed by `preprocess.ipynb`, is in the csv file `responses.csv`. Stimuli are indexed by column numbers. The corresponding stimuli are given in the `pair` columns. For example, a response in column `0` indicates that the participant was responding to the stimulus shown in the column `pair0`. 

# Replicating the analysis
The results reported in the paper can be replicated by running the `replicate.ipynb` notebook.

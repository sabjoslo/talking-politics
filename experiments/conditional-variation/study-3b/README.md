Materials for analysis of study 3b reported in "Two kinds of discussion and types of conversation".

# Preprocessing the data
The participant data downloaded from Qualtrics is in the csv file `LoP_WordGroups.csv`. This study was run in conjunction with an unrelated study on perceptual decision-making. Data columns concerning that study and response ID columns have been removed. The cleaned version, processed by `preprocess.ipynb`, is in the csv file `responses.csv`. Stimuli are indexed by column numbers. The corresponding stimuli are given in the `list` columns. For example, a response in column `1` indicates that the participant was responding to the stimulus shown in the column `list1`, which is associated with the party shown in the column `list1_party`. 

Valence ratings were collected for each item in a separate study. The participant data from this Qualtrics study is in the csv file `LoP_ValenceRatings.csv` (response ID columns have been removed). The cleaned version, processed by `preprocess.ipynb`, is in the csv file `valence_data.csv`.

# Replicating the analysis
The results reported in the paper can be replicated by running the `replicate.ipynb` notebook.

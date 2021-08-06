Contains the materials needed to replicate the analyses of Study 3'.

- `demographics.csv` contains the cleaned demographic data.
  - `ID` contains unique participant identifiers that link data from the same participant across surveys.
  - `gender_strength` contains the participant's response to the question "How strongly do you identify with your gender?," where "1" indicates "very little" and "7" indicates "very much."
  - `party_strength` contains the participant's response to the question "How strongly do you identify with your party?," where "1" indicates "very little" and "7" indicates "very much."
  - `political_bubble` contains the participant's response to the question "What percent of your close friends and family share your political affiliation?"
- `valence.csv` contains the cleaned participant data from the Valence Survey.
  - Columns `100` - `199` contain the word the participant chose when presented with the corresponding antonym pair (these pairs are listed in the corresponding order in `stimuli.csv`; for example, column `100` contains choices between the words in the first row of `stimuli.csv` (*food / foods*)).
  - `Condition` indicates the condition the participant was assigned to (in this case, whether they were asked to select the word that struck them as a more positively-valenced, or asked to the select the word that struck them as more negatively-valenced).
  - `atc_passed` indicates whether the participant passed the attention check. Participants for whom this value is "FALSE" were not included in reported analyses.
  - `order` indicates the order in which the participant was presented with the Valence Survey. For example, the participant in the first row responded to the Valence Survey second.
- `politics.csv` contains the cleaned participant data from the Politics Survey. Columns are defined analogously to those in `valence.csv`. `ident` indicates the participant's self-reported political affiliation.
- `cl.csv` contains the cleaned participant data from the Construal Survey. Columns are defined analogously to those in `valence.csv`.
- `gender.csv` contains the cleaned participant data from the Gender Survey. Columns are defined analogously to those in `valence.csv`. `ident` indicates the participant's self-reported gender identity.
- `stimuli.csv` contains the list of stimuli. The `valence` columns are the mean ratings in the data collected and made available by [Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods, 45.*](dx.doi.org/10.3758/s13428-012-0314-x). The `sig` columns contain the words' partial Kullback-Leibler divergence from *P(D)* to *P(R)* (see Appendix A).
- `replicate.ipynb` replicates the pre-registered analyses and supplemental analyses included in the paper.

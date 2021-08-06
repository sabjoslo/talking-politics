Contains the materials needed to replicate the analyses of Study 2.

- `responses.csv` contains the cleaned participant data.
  - Columns `1` - `10` contain the word the participant chose when presented with the corresponding antonym pair (these pairs are listed in the corresponding order in Table B3 of the paper; for example, column `1` contains choices between the words in the first row of Table B3 (*superior / inferior*)).
  - `Condition` indicates whether the participant was asked to select the word more likely to have been spoken by a Democrat or by a Republican.
  - `party_strength` contains the participant's response to the question "How strongly do you identify as a `party_affil`?," where "1" indicates "very little" and "7" indicates "very much."
  - `political_engagement` contains the participant's response to the question "How engaged are you in politics?," where "1" indicates "not at all" and "7" indicates "extremely."
  - `imc_passed` indicates whether the participant passed the instructional manipulation check (Oppenheimer, Meyvis, & Davidenko, 2009). Participants for whom this value is "FALSE" were not included in reported analyses.
- `replicate.ipynb` replicates the analyses in the paper.
- `fig2b.py` reproduces Figure 2B.

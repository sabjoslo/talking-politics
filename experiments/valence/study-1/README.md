Contains the materials needed to replicate the analyses of Study 1.

- `responses.csv` contains the cleaned participant data.
  - Columns `1` - `10` contain the participant rating of the word presented in the corresponding `pair` column. "1" indicates a response of "I am almost certain the speaker is a Democrat," and "6" indicates a response of "I am almost certain the speaker is a Republican." For example, the participant in the first row was presented the word "sorrow" (and not the word "joy") and gave a response corresponding to 2 ("I am reasonably sure the speaker is a Democrat").
  - `party_strength` contains the participant's response to the question "How strongly do you identify as a `party_affil`?," where "1" indicates "very little" and "7" indicates "very much."
  - `political_engagement` contains the participant's response to the question "How engaged are you in politics?," where "1" indicates "not at all" and "7" indicates "extremely."
  - `imc_passed` indicates whether the participant passed the instructional manipulation check (Oppenheimer, Meyvis, & Davidenko, 2009). Participants for whom this value is "FALSE" were not included in reported analyses.
- `replicate.ipynb` replicates the analyses in the paper.
- `fig1b.py` reproduces Figure 1B.

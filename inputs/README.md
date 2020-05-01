Files in inputs.

- `crec_w2v`: Word2vec model trained on the CongRec corpus. Used to generate stimuli for study 2.
- `legislators-current.json`: List of current members of Congress. Downloaded from https://github.com/unitedstates/congress-legislators. Used to exclude words during pre-processing of the corpora.
- `legislators-historical.json`: List of historical members of Congress. Downloaded from https://github.com/unitedstates/congress-legislators. Used to exclude words during pre-processing of the corpora.
- `list_of_countries.html`: HTML source of https://www.state.gov/countries-areas/. Used to exclude words during pre-processing of the corpora.
- `moderators_fullnames`: Manually-coded names of moderators of the U.S. Presidential debates. Used to exclude words during pre-processing of the corpora.
- `participants_fullnames`: Manually-coded names of participants in the U.S. Presidential debates. Used to exclude words during pre-processing of the corpora.
- `relative_metrics-bigrams`: Pandas DataFrame containing information about the relative frequencies of occurrences of bigrams. Input for `procedural_words.py`.
- `stop-edited.txt`: Stopwords listed at http://snowball.tartarus.org/algorithms/english/stop.txt, lightly edited for formatting. Used to exclude words during pre-processing of the corpora.
- `table9`: Stand-in for raw text file of words in Table 9 of Gentzkow, M., Shapiro, J.M. and Taddy, M. (2019). Measuring Group Differences in High‐Dimensional Choices: Method and Application to Congressional Speech. Available at https://doi.org/10.3982/ECTA16566. Used to exclude words during pre-processing of the corpora.
- `us_cities.html`: HTML source of https://www.biggestuscities.com/. Used to exclude words during pre-processing of the corpora.
- `usstates`: List of states from liststates.com. Used to exclude words during pre-processing of the corpora.
- `words-unigrams_proceduraladded.csv`: Output of procedural word coding (see repository README). Used to exclude words during pre-processing of the corpora.

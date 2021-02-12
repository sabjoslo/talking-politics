A library to derive information-theoretic metrics and perform other textual analysis on political speech from the Congressional Record and U.S. presidential debates.

Includes code and data used in preparation of

Sloman SJ, Oppenheimer DM, DeDeo S (2021) Can we detect conditioned variation in political speech? two kinds of discussion and types of conversation. PLoS ONE 16(2): e0246689. https://doi.org/10.1371/journal.pone.0246689

and

Sloman, S.J., Oppenheimer, D., and DeDeo, S. (2020). One fee, two fees; red fee, blue fee: People use the valence of others' speech in social relational judgments. *Under review.*

If you have questions, or notice that anything needed to replicate the analysis is missing, please contact Sabina Sloman at ssloman@andrew.cmu.edu.

# Configuration

User-specific configuration settings can be specified in the `settings` file.

# Getting and pre-processing data

## Congressional Record

1. Use the [congressional-record project](https://github.com/unitedstates/congressional-record) to download data from <https://www.gpo.gov/fdsys/>.
2. `./json_to_text.py --mode=speech` converts the raw JSON data into plain text.

## Presidential debates

1. `python -c "from pyspan import debates; debates.get_debates()"` downloads transcripts of the presidential debates from [The American Presidency Project](http://www.presidency.ucsb.edu/debates.php) and converts them to JSON format.
2. `python -c "from pyspan import debates; debates.get_participants(); debates.get_moderators()"` creates hand-coded lists of the participants and moderators of the debates. This information helps the text parser separate the speaker from what they say.
3. `python -c "from pyspan import debates; debates.get_text_from_debates()"` converts the JSON-formatted data to plain text files.

## Excluding words

`python -c "from pyspan import purger; the_purger = purger.Purger(); the_purger.stopwords"` will display the words excluded from the counts (below). The exclusion criteria for words are detailed in "Two kinds of discussion and types of conversation."

### Classifying words as procedural

Procedural words were selected based on the following criteria:
1. Contained in a bigram that's said in more than 1% of speeches,
2. Contained in a bigram that's said more than 10x as much as in general speech, and
3. The word before and after it are the same in the majority of contexts seen.

Running `pyspan/scripts/get_materials_for_procedural_words_coding.py` creates `output/words-unigrams.csv`, a csv with the unigrams that satisfy criteria 1 & 2, and `output/excerpts-unigrams.tex`, which can be compiled into a PDF with three randomly-selected excerpts that contain each of these unigrams. This PDF was used by an RA to code whether or not the word was preceded and followed by the same word in at least two of these instances, in which case, they coded it as procedural in `inputs/words-unigrams-proceduraladded.csv`.

# Calculating information-theoretic metrics

The `mode` settings (configurable in `settings`) determines whether or not these metrics are calculated for the Congressional Record or presidential debates data.

1. `python -c "from pyspan import count_words; count_words.count_unigrams()"` converts the plain text data into counts of unigrams or bigrams.
2. `python -c "from pyspan import bitcounters; bitcounters.BitCounter(mode = 'unigrams').get_all()"` converts these count files into information-theoretic measurements.

# Study design and analysis

The directory `experiments` contains the raw data and Jupyter notebook files used to analyze these data.

The contents of this repository are licensed under a [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/).

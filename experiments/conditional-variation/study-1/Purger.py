from bs4 import BeautifulSoup
from itertools import chain
import json
import os
import pandas as pd
import pickle
import re
import string
from pyspan.config import *
indir = paths["input_dir"]
from pyspan.utils import *

# The data I have of legislators' names includes non-ASCII characters, while
# all the CRec data has been converted to ASCII somewhere up the pipeline. This
# lookup can be used to convert these non-ASCII characters to the character
# used when these legislators are referred to in the plain text CRec data. Add
# more characters as needed.
xml_to_ascii={ "&#225;": "a", "&#243;": "o", "&#233;": "e", "&#8217;": "'" }

class Purger(object):
    def __init__(self, years=range(2012, 2018), overwrite=False):
        self.years_=map(str, years)
        self.stopwords=self.get_general_stopwords()
        self.stopwords+=self.get_congressional_stopwords()
        self.stopwords+=self.get_places()
        self.get_names_of_legislators(overwrite=overwrite)
        names=set(self.rep_names["lastname"].values)
        names=chain(*[ re.sub('[%s]'%re.escape(string.punctuation), ' ',
                       self.to_ascii(name.lower())).split() for name in names ])
        self.stopwords+=list(names)
        self.stopwords=list(set(self.stopwords))

    def to_ascii(self, s):
        s=s.encode("ascii", "xmlcharrefreplace")
        for k, v in xml_to_ascii.items():
            s=s.replace(k, v)
        return s

    # Gentzkow, Matthew, Shapiro, Jesse M. and Taddy, Matt. "Congressional
    # Record for the 43rd-114th Congresses: Parsed Speeches and Phrase Counts".
    #
    # Eliminate stopwords from the list used by Gentzkow, Shapiro & Taddy
    # (2017): http://snowball.tartarus.org/algorithms/english/stop.txt
    def get_general_stopwords(self):
        stop=[]
        with open(indir + "stop-edited.txt", "r") as rfh:
            for line in rfh:
                word=line.split("|")[0].strip()
                if word:
                    stop+=re.sub('[%s]'%re.escape(string.punctuation), ' ', word).split()
        return stop

    # iii. The last names of congresspeople
    # Data from https://github.com/unitedstates/congress-legislators
    def get_names_of_legislators(self, overwrite=False):
        if os.path.exists(indir + "rep-names.pkl") and not overwrite:
            with open(indir + "rep-names.pkl", "rb") as rfh:
                self.rep_names=pickle.load(rfh)
            if os.path.exists(indir + "sen-names.pkl") and not overwrite:
                with open(indir + "sen-names.pkl", "rb") as rfh:
                    self.sen_names=pickle.load(rfh)
                return
        for f in (indir + "legislators-historical.json",
                  indir + "legislators-current.json"):
            with open(f, "r") as rfh:
                jobj=json.load(rfh)
            rep_dat=[]
            sen_dat=[]
            for obj in jobj:
                for term in obj["terms"]:
                    if ( term["start"].split("-")[0] not in self.years_ and
                         term["end"].split("-")[0] not in self.years_ ):
                        continue
                    start=timestamp_to_integer(term["start"], "%Y-%m-%d")
                    end=timestamp_to_integer(term["end"], "%Y-%m-%d")
                    if term["type"]=="rep":
                        rep_dat.append((start, end, obj["name"]["last"]))
                    elif term["type"]=="sen":
                        sen_dat.append((start, end, obj["name"]["last"]))

        rep_df=pd.DataFrame(rep_dat, columns=[ "start", "end", "lastname" ])
        sen_df=pd.DataFrame(sen_dat, columns=[ "start", "end", "lastname" ])
        with open(indir + "rep-names.pkl", "wb") as wfh:
            pickle.dump(rep_df, wfh)
        with open(indir + "sen-names.pkl", "wb") as wfh:
            pickle.dump(sen_df, wfh)
        self.rep_names=rep_df
        self.sen_names=sen_df
        return

    # "...bigrams containing the stem of a US-Congress-specific stopword are
    # flagged. Stopwords come from three sources: (i) the manually selected
    # stopwords in Table 9, (ii) the names of states, and (iii) the last names
    # of all congresspeople recorded in the historical source."
    def get_congressional_stopwords(self):
        # i. Table 9 in GST
        with open(indir + "table9", "r") as rfh:
            cong_stopwords=list(chain(*(line.split() for line in rfh.read().split("\n"))))
        assert len(cong_stopwords)==93

        # ii. Names of states (from liststates.com)
        with open(indir + "usstates", "r") as rfh:
            states=[ line.lower() for line in rfh.read().split() ]

        # Federal district and inhabited territories (from
        # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States)
        territories=[ "district", "columbia", "samoa", "guam", "northern",
                      "mariana", "islands", "puerto", "rico", "us", "virgin" ]
        cong_stopwords+=states
        cong_stopwords+=territories

        return cong_stopwords

    # Include the names of countries and major US cities
    def get_places(self):
        # Countries
        # Data from https://www.state.gov/misc/list/index.htm
        with open(indir + "list_of_countries.html", "r") as rfh:
            text=rfh.read()
        soup=BeautifulSoup(text)
        countries=[ el.text for el in soup.findAll("a") if el.has_attr("href")
                    and not
                    isinstance(re.match("http://www.state.gov/p/[a-z]*/ci/[a-z]*",
                    el.attrs["href"]), type(None)) ]
        countries=chain(*[ re.sub('[%s]'%re.escape(string.punctuation), ' ',
                           country.lower()).split() for country in countries ])

        # Cities
        # Data from https://www.biggestuscities.com/
        with open(indir + "us_cities.html", "r") as rfh:
            text=rfh.read()
        soup=BeautifulSoup(text)
        cities=[]
        for tr in soup.findAll("tr"):
            tds=tr.findAll("td")
            if len(tds)<2:
                continue
            rank, city=tds[:2]
            try:
                if int(rank.text)>100:
                    continue
            except ValueError:
                continue
            cities.append(city.text.strip())
        cities=chain(*[ re.sub('[%s]'%re.escape(string.punctuation), ' ',
                        city.lower()).split() for city in cities ])

        return list(set(list(countries)+list(cities)))

    def get_stopwords(self, chamber, timestamp):
        # Add in names of legislators serving in this chamber on this date
        # TODO: How am I dealing with non-ASCII characters?
        time_int=timestamp_to_integer(timestamp, "%Y-%m-%d")
        self.get_legislators_names()
        if chamber=="rep":
            names=list(set(self.rep_names[self.rep_names["start"]<=time_int][self.rep_names["end"]>=time_int]["lastname"]))
        elif chamber=="sen":
            names=list(set(self.sen_names[self.sen_names["start"]<=time_int][self.sen_names["end"]>=time_int]["lastname"]))
        self.stopwords+=names

    def purge(self, text):
        return " ".join(word for word in text.split() if word not in
                        self.stopwords)

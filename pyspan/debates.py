from bs4 import BeautifulSoup, element
from collections import defaultdict
from itertools import chain
import json
import os
import numpy as np
import pandas as pd
import pickle
import re
import requests
from scipy import stats
import string
import time
from wordplay.utils import to_ascii
from pyspan.config import *
input_dir = paths["input_dir"]
proc_txt_dir = paths["proc_txt_dir"]
raw_data_dir = paths["raw_data_dir"]
cycles = sorted(settings["election_cycles"], reverse = True)
types_ = [ t for t in ("general", "primary") if t in settings["debate_types"] ]

BASE_URL = "http://www.presidency.ucsb.edu/debates.php"

def get_(url):
    print "Sending request to {}.".format(url)
    return requests.get(url, headers = headers)

def clean(data, d):
    # Clean data elements
    content = []
    re_reaction = r"((?<=\[)(<i>)?[\w ]*(</i>)?(?=\]))"
    for dat in data:
        speaker = BeautifulSoup(dat[0]).get_text().rstrip(":")
        reactions = [ match[0] for match in re.findall(re_reaction, dat[1]) ]
        re_reaction_ = r"\[(?:{})\]".format("|".join(reactions))
        speech = re.split(re_reaction_, dat[1])
        assert len(reactions) == len(speech)-1
        section = zip(speech, reactions + [""])
        section = chain(*section)
        section = map(lambda s: BeautifulSoup(s).get_text(), section)
        section = [ s.strip() for s in section ]
        for i, s in enumerate(section):
            if not s.strip():
                continue
            if i%2 == 0:
                content.append({speaker.rstrip(":"): s})
                continue
            content.append({None: s.upper()})
    d["content"] = content

    return d

# Standardize participant names
def _standardize_names(s, participants):
    if participants:
        for p in participants:
            if p in s.lower():
                s = p
                break
    return s

# The general Presidential debates from the 2012 cycle are in a weird format
def parse_2012_general_debate(d, text, participants_ = None):
    # Get debate metadata
    participants = [ "Barack Obama (D)", "Mitt Romney (R)" ]

    re_ = r"(?<=<i>Moderator )[\w ]*(?=\.</i>|</i>\.)"
    moderator = re.search(re_, text).group()
    d["moderator"] = moderator

    re_first_speaker = r"(?:<i>)*(Moderator {}|M[rs]\. [\w]*|The President|Republican Presidential Nominee W\. Mitt Romney|Gov\. Romney|Q)(?:\. ?</i>|</i> ?\.|\. )".format(moderator)
    re_speaker = r"</p><p>" + re_first_speaker
    first_speaker = re.search(re_first_speaker, text)
    data = [ (text[first_speaker.start(0):first_speaker.end(0)],
              text[first_speaker.end(0):next(re.finditer(re_speaker, text)).start(0)]
              + "</p><p>") ]
    data += zip(map(lambda s: re.sub(r"[. ]*$", "", s), re.findall(re_speaker, text)),
                map(lambda s: re.sub(r"^[. ]*", "", s) + "</p><p>",
                    re.split(re_speaker, text)[2::2]))

    re_headings = r"(<p>(<i> *)*[\w \-'/,.]+\w( *</i>)*</p><p>)"

    def sub_(s):
        matches = re.findall(re_headings, s)
        s_ = ""
        if len(matches) > 0:
            s_ = matches[0][0]
        return s.replace(s_, "")

    def name_(s):
        s = re.sub(r"Moderator {}".format(moderator), "MODERATOR", s)
        s = re.sub(r"Mr\. Romney", "romney", s)
        s = re.sub(r"M[rs]. [\w]*", "MODERATOR", s)
        s = re.sub(r"The President", "obama", s)
        s = re.sub(r"Republican Presidential Nominee W. Mitt Romney", "romney", s)
        s = re.sub(r"Gov. Romney", "romney", s)
        return s

    data = map(lambda d_: (name_(d_[0]), sub_(d_[1])), data)

    # Clean data elements
    return clean(data, d)

# Generic parser
def parse(d, text, participants_ = None):
    # Get metadata and speakers
    re_speaker = r"<b> *[A-Z,' ]*(?: \[<i>through translator</i>\])?(?:</b>)?:(?: *</b>)?"
    data = zip(map(lambda s: _standardize_names(s, participants_),
                   re.findall(re_speaker, text)), re.split(re_speaker, text)[1:])

    def _extract_names(re_):
        names = [ dat[1] for dat in data if re.match(re_, dat[0]) ]
        if len(names) > 0:
            assert len(names) == 1
            names = names[0]
            names = re.split(r";? ?(?: ?and ?)?<br/>", names)
            names = [ p for p in names if isinstance(p, str) and p.strip() ]
            names = map(lambda p: p.strip(), names)
            names = [ p for p in names if p != "and" ]
            names = map(lambda p: BeautifulSoup(p).get_text(), names)
            return [ p for p in names if p.strip() ]
        return None

    # Get debate metadata
    re_ = r"<b> *PARTICIPANTS: *</b>"
    participants = _extract_names(re_)
    if participants:
        d["participants"] = participants
    data = [ dat for dat in data if not re.match(re_, dat[0]) ]

    re_ = r"<b> *PANELISTS: *</b>"
    panelists = _extract_names(re_)
    if panelists:
        d["panelists"] = panelists
    data = [ dat for dat in data if not re.match(re_, dat[0]) ]

    re_ = r"<b> *SPONSOR: *</b>"
    sponsor = _extract_names(re_)
    if sponsor:
        d["sponsor"] = sponsor
    data = [ dat for dat in data if not re.match(re_, dat[0]) ]

    # Moderator data is slightly different, so it gets its own code block.
    re_ = r"<b> *MODERATORS?: *</b>"
    moderator_ = [ dat[1] for dat in data if re.match(re_, dat[0]) ]
    if len(moderator_) > 0:
        moderator_raw_str = moderator_[0]
        moderator = re.split(r"(;)?( ?and ?)?<br/>", moderator_raw_str)
        moderator = [ m for m in moderator if isinstance(m, str) and m.strip() ]
        moderator = map(lambda m: m.strip(), moderator)
        moderator = [ m for m in moderator if
                      isinstance(re.match(r"(;| ?and ?)", m), type(None)) ]
        moderator = map(lambda m: BeautifulSoup(m).get_text(), moderator)
        moderator = [ m for m in moderator if m.strip() ]
        d["moderator"] = moderator
        data = [ dat for dat in data if not
                 re.match(re.escape(moderator_raw_str), dat[1]) ]

    # Clean data elements
    return clean(data, d)

def parse_raw_text(soup, cycle, type_, party, kind, date, fmt_date, save = True,
                   participants_d = None):
    assert settings["mode"] == "debates"

    # Is this a vice presidential debate?
    span = soup.findAll("span", class_ = "paperstitle")[0]
    vice_presidential = "Vice " in span.text
    which_office = "N/A"
    if type_ == "general":
        which_office = "vice" if vice_presidential else "president"

    # Get the text of the debate
    span = soup.findAll("span", class_ = "displaytext")[0]
    text = str(span)
    s = re.search(r"<span class=\"displaytext\">", text).group()
    text = text.replace(s, "")
    s = re.search(r"<hr noshade[\w\W]*", text).group()
    text = text.replace(s, "")

    d = dict()
    d["cycle"] = cycle
    d["type"] = type_
    d["party"] = party if party != "N/A" else None
    d["kind"] = kind if kind != "N/A" else None
    d["office"] = which_office if which_office != "N/A" else None
    d["date"] = date

    if participants_d:
        quals = [ item_ for item_ in (str(cycle), type_, party, which_office,
                  kind, fmt_date) if item_ != "N/A" ]
        quals = tuple(quals)
        participants_ = participants_d[quals]
    else:
        participants_ = None

    if cycle == 2012 and type_ == "general" and not vice_presidential:
        d = parse_2012_general_debate(d, text, participants_)
    else:
        d = parse(d, text, participants_)

    # Convert to JSON
    if save:
        dir_ = "{}{}/{}/{}{}{}".format(raw_data_dir, cycle, type_,
               which_office + "/" if which_office != "N/A" else "", party + "/"
               if party != "N/A" else "", kind + "/" if kind != "N/A" else "")
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        fn = "{}.json".format(dir_ + fmt_date)
        with open(fn, "w") as wfh:
            json.dump(d, wfh)

def helpful_children(element_, generations = 1):
    children = element_.children if generations == 1 else element_.descendants
    return [ child for child in children if not isinstance(child,
             element.NavigableString) and child.text.strip() ]

def get_debates():
    assert settings["mode"] == "debates"

    # If it exists, get list of participants in each debate
    participants_fn = input_dir + "participants"
    if os.path.exists(participants_fn):
        participants_ = pickle.load(open(participants_fn, "rb"))
    else:
        participants_ = None

    # Get a list of the webpages for all debates
    response = get_(BASE_URL)
    soup = BeautifulSoup(response.content)

    tbody = [ t for t in soup.findAll("tbody") if
              t.text.strip()[:len("2016")] == "2016" ][1]
    children = helpful_children(tbody)

    debates = []

    i = 0
    for cycle in cycles:
        while str(cycle) not in children[i].text:
            i += 1
        child = children[i]
        for type_ in types_:
            type_str = type_[0].upper() + type_[1:] + " Election"
            while type_str not in children[i].text:
                i += 1
            i += 1
            child = children[i]

            # Get headings indicating main/undercard debates
            def _is_undercard_debate(kind, i):
                l = helpful_children(children[i], generations = "all")
                if any([ ["pppusdate"] in l[i_].attrs.values() for i_ in
                         range(len(l)) ]):
                    kind = children[i].text.split()[0].replace("\"", "")
                    i += 1
                return kind, i

            n_iter = 2 if type_ == "primary" else 1
            for j in range(n_iter):
                # When the incumbent's running, there's only one set of primary
                # debates
                if j == 1 and " Party" not in child.text:
                    break
                party, kind = "N/A", "N/A"
                if type_ == "primary":
                    party = child.text.split()[0].lower()
                    kind, i = _is_undercard_debate(kind, i+1)
                    child = children[i]
                date = child.findAll("td", class_ = "docdate")
                text = child.findAll("td", class_ = "doctext")
                while len(date) > 0 and len(text) > 0:
                    date, text = date[0], text[0]
                    debates.append((text.a.attrs["href"], cycle, type_, party,
                                    kind, date.text))
                    kind, i = _is_undercard_debate(kind, i+1)
                    child = children[i]
                    date = child.findAll("td", class_ = "docdate")
                    text = child.findAll("td", class_ = "doctext")

    # Get the text for each debate
    for url, cycle, type_, party, kind, date in debates:
        response = get_(url)
        soup = BeautifulSoup(response.content)
        suf = re.findall(r"(?<=[0-9])(st,|nd,|rd,|th,|,)", date)[0]
        fmt_date = time.strptime(date, "%B %d{} %Y".format(suf))
        fmt_date = time.strftime("%m_%d_%y", fmt_date)
        parse_raw_text(soup, cycle, type_, party, kind, date, fmt_date,
                       participants_d = participants_)

def get_text_from_debates():
    assert settings["mode"] == "debates"

    def _process(s):
        s = re.sub('[%s]'%re.escape(string.punctuation),' ', s.lower())
        return re.sub(' +', ' ', s)

    participants = pickle.load(open(input_dir + "participants", "rb"))

    for cycle in cycles:
        dem_content, repub_content = [], []
        for type_ in types_:
            base_dir = "{}{}/{}/".format(raw_data_dir, cycle, type_)
            if type_ == "primary":
                dem_dir, repub_dir = base_dir + "democratic", base_dir + "republican"
                dem_files = chain(*[ [ "{}/{}".format(dir_[0], f) for f in
                                       dir_[2] if f != ".DS_Store" ] for dir_ in
                                       os.walk(dem_dir) ])
                repub_files = chain(*[ [ "{}/{}".format(dir_[0], f) for f in
                                         dir_[2] if f != ".DS_Store" ] for dir_
                                         in os.walk(repub_dir) ])

                for files, content in [ (dem_files, dem_content), (repub_files,
                                                                   repub_content) ]:
                    for f in files:
                        fmt_date = f.split("/")[-1].rstrip(".json")
                        json_obj = json.load(open(f, "r"))
                        cycle = json_obj["cycle"]
                        type_ = json_obj["type"]
                        party = json_obj["party"]
                        office = json_obj["office"]
                        kind = json_obj["kind"]
                        quals = [ item_ for item_ in (str(cycle), type_, party,
                                  office, kind, fmt_date) if item_ ]
                        quals = tuple(quals)
                        participants_ = participants[quals]
                        for speech in json_obj["content"]:
                            speaker, speech_ = speech.items()[0]
                            speech_ = _process(speech_)
                            # Remove moderator from content
                            if speaker not in participants_ or not speech_.strip():
                                continue
                            content.append((speaker, speech_, cycle, type_, party,
                                            office, kind, fmt_date))

            elif type_ == "general":
                files = chain(*[ [ "{}/{}".format(dir_[0], f) for f in dir_[2]
                                   if f != ".DS_Store" ] for dir_ in
                                   os.walk(base_dir) ])
                for f in files:
                    fmt_date = f.split("/")[-1].rstrip(".json")
                    json_obj = json.load(open(f, "r"))
                    cycle = json_obj["cycle"]
                    type_ = json_obj["type"]
                    office = json_obj["office"]
                    kind = json_obj["kind"]
                    quals = [ item_ for item_ in (str(cycle), type_, office,
                              kind, fmt_date) if item_ ]
                    quals = tuple(quals)
                    participants_ = participants[quals]
                    for speech in json_obj["content"]:
                        speaker, speech_ = speech.items()[0]
                        speech_ = _process(speech_)
                        # Remove moderator from content
                        if speaker not in participants_ or not speech_.strip():
                            continue
                        if speaker in ("obama", "biden", "clinton", "kaine"):
                            party, content = "democratic", dem_content
                        elif speaker in ("romney", "ryan", "trump", "pence"):
                            party, content = "republican", repub_content
                        content.append((speaker, speech_, cycle, type_, party,
                                        office, kind, fmt_date))

        ddf = pd.DataFrame(dem_content, columns = [ "speaker", "speech", "cycle",
                                                    "type", "party", "office",
                                                    "kind", "date" ])
        rdf = pd.DataFrame(repub_content, columns = [ "speaker", "speech", "cycle",
                                                      "type", "party", "office",
                                                      "kind", "date" ])

        # Save dataframes
        with open("{}{}/dem_df".format(proc_txt_dir, cycle), "wb") as wfh:
            pickle.dump(ddf, wfh)
        with open("{}{}/repub_df".format(proc_txt_dir, cycle), "wb") as wfh:
            pickle.dump(rdf, wfh)

        # Save as txt so count_words can find it
        for df_, fn in [ (ddf, "dem_speech"), (rdf, "repub_speech") ]:
            df = df_.replace([None], "N/A")
            for cycle_ in np.unique(df["cycle"]):
                for type__ in np.unique(df["type"]):
                    for party in np.unique(df["party"]):
                        for office in np.unique(df["office"]):
                            for kind in np.unique(df["kind"]):
                                for date in np.unique(df["date"]):
                                    speech = df.loc[(df["cycle"] == cycle_) &
                                                    (df["type"] == type__) &
                                                    (df["party"] == party) &
                                                    (df["office"] == office) &
                                                    (df["kind"] == kind) &
                                                    (df["date"] == date)]["speech"]
                                    if len(speech) > 0:
                                        dir_ = proc_txt_dir + "".join(str(qual) + "/"
                                                                      for qual in
                                                                      (cycle_,
                                                                       type__,
                                                                       party,
                                                                       office,
                                                                       kind,
                                                                       date) if
                                                                      qual != "N/A")
                                        if not os.path.exists(dir_):
                                            os.makedirs(dir_)
                                        with open(dir_ + fn, "w") as wfh:
                                            wfh.write("\n".join(map(lambda s:
                                                s.encode("ascii", "ignore"),
                                                speech.values)))

# A method to elicit (and, optionally, save) a list of hand-coded names of
# participants. This method was used to generate a file of participant last
# names, saved as "participants", in order to isolate participant speech when
# get_text_from_debates() is called. It was also used to generate a file of
# participant first and last names, saved as "participants_fullnames", to be
# added to pyspan.Purger.Purger's list of stopwords.
def get_participants(fn = "participants"):
    assert settings["mode"] == "debates"

    files = chain(*[ [ "{}/{}".format(dir_[0], f) for f in dir_[2] if f !=
                       ".DS_Store" ] for dir_ in os.walk(raw_data_dir) ])
    participants = dict()
    ps_lu = dict()
    for f in files:
        quals = f.lstrip(raw_data_dir).rstrip(".json")
        quals = tuple(quals.split("/"))
        json_obj = json.load(open(f, "r"))
        if "participants" not in json_obj.keys():
            ps = raw_input("Participants for debate\n{}:".format("\n".join(quals)))
            ps = ps.split()
        else:
            ps_, ps = json_obj["participants"], []
            for p in ps_:
                if p in ps_lu:
                    ps.append(ps_lu[p])
                else:
                    tp = raw_input("Code for participant {}".format(p))
                    ps.append(tp)
                    ps_lu[p] = tp
        participants[quals] = ps
    if fn:
        with open(input_dir + fn, "wb") as wfh:
            pickle.dump(participants, wfh)

# A method to elicit (and, optionally, save) a list of hand-coded names of
# moderators. This method was used to generate a file of moderator first names,
# last names and other non-standard English strings associated with the
# moderator (e.g. network name) saved as "moderators_fullnames", to be
# added to pyspan.Purger.Purger's list of stopwords.
def get_moderators(fn = "moderators"):
    assert settings["mode"] == "debates"

    files = chain(*[ [ "{}/{}".format(dir_[0], f) for f in dir_[2] if f !=
                       ".DS_Store" ] for dir_ in os.walk(raw_data_dir) ])
    moderators = dict()
    #moderators = pickle.load(open(input_dir + fn, "rb"))
    ms_lu = dict()
    for f in files:
        quals = f.lstrip(raw_data_dir).rstrip(".json")
        quals = tuple(quals.split("/"))
        #if quals in moderators:
        #    continue
        json_obj = json.load(open(f, "r"))
        if "moderator" not in json_obj.keys():
            ms = raw_input("Moderator for debate\n{}:".format("\n".join(quals)))
            ms = ms.split()
        else:
            ms_, ms = json_obj["moderator"], []
            if isinstance(ms_, basestring):
                ms_ = [ms_]
            for m in ms_:
                if m in ms_lu:
                    ms.append(ms_lu[m])
                else:
                    tm = raw_input("Code for participant {}".format(to_ascii(m)))
                    ms.append(tm)
                    ms_lu[m] = tm
        moderators[quals] = ms
    if fn:
        with open(input_dir + fn, "wb") as wfh:
            pickle.dump(moderators, wfh)

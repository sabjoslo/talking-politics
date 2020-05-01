#!/usr/bin/python2.7

import json
import logging
import lxml.etree
import os
import pandas as pd
import pickle
import re
import shlex
import string
import sys
import time
import subprocess
from pyspan.config import *
indir = paths["input_dir"]
log_dir = paths["log_dir"]
PROC_TXT_DIR = paths["proc_txt_dir"]
RAW_DATA_DIR = paths["raw_data_dir"]
years = settings["years"]
from pyspan.utils import *

# https://github.com/propublica/Capitol-Words/blob/4e02e594feee0018a740513694d40efffa59808c/parser/parser.py
re_speaking = r'^ *((((Mr)|(Ms)|(Mrs)|(Miss))\.? [A-Za-z \- \']+(of [A-Z][a-z]+)?)|((The (VICE |Acting |ACTING )?(PRESIDENT|SPEAKER)( pro tempore)?)|(The PRESIDING OFFICER)|(The CLERK)))\.'
re_address = r'^ *m(r|(adam)).? speaker,?'
_re_ln_helper='[A-Z\-\'( (?!of))]+[a-z]?[A-Z\-\'( (?!of))]*(?=( of [A-Za-z ]+)?\.)'
re_lastname = r'((?<=Mr\. )%(h)s)|((?<=Ms\. )%(h)s)|((?<=Mrs\. )%(h)s)|((?<=Miss )%(h)s)'%{'h':_re_ln_helper}

# ridgerunner, you are a lifesaver. The expression below is adapted from the
# code found at https://stackoverflow.com/questions/8465335/#8466725
re_sent_end = re.compile(r"""
        # Split sentences on whitespace between them.
        (?:               # Group for two positive lookbehinds.
          (?<=[.!?])      # Either an end of sentence punct,
        | (?<=[.!?]['"])  # or end of sentence punct and quote.
        | (?<=[.!?]'')
        )                 # End group of two positive lookbehinds.
        # Only care about abbrevs. when they're standalone (i.e. not at the ends
        # of words, e.g. "This is a sentence about SALT.").
        (?<!  [\s(]Mr\.     )    # Don't end sentence on "Mr."
        (?<!  ^Mr\.      )
        (?<!  [\s(]Mrs\.    )    # Don't end sentence on "Mrs."
        (?<!  ^Mrs\.     )
        (?<!  [\s(]Ms\.     )    # Don't end sentence on "Ms."
        (?<!  ^Ms\.      )
        (?<!  [\s(]Jr\.     )    # Don't end sentence on "Jr."
        (?<!  ^Jr\.      )
        (?<!  [\s(]Dr\.     )    # Don't end sentence on "Dr."
        (?<!  ^Dr\.      )
        (?<!  [\s(]Prof\.   )    # Don't end sentence on "Prof."
        (?<!  ^Prof\.    )
        (?<!  [\s(]Sr\.     )    # Don't end sentence on "Sr."
        (?<!  ^Sr\.      )
        (?<!  [\s(]St\.     )    # Don't end sentence on "St."
        (?<!  ^St\.      )
        (?<!  [\s(]Rd\.     )    # Don't end sentence on "Rd."
        (?<!  ^Rd\.      )
        (?<!  [\s(]Ave\.    )    # Don't end sentence on "Ave."
        (?<!  ^Ave\.     )
        (?<!  [\s(]Blvd\.   )    # Don't end sentence on "Blvd."
        (?<!  ^Blvd\.    )
        (?<!  [\s(]U\.S\.   )    # Don't end sentence on "U.S."
        (?<!  ^U\.S\.    )
        (?<!  [\s(]a\.m\.   )    # Don't end sentence on "a.m."
        (?<!  ^a\.m\.    )
        (?<!  [\s(]p\.m\.   )    # Don't end sentence on "p.m."
        (?<!  ^p\.m\.    )
        (?<!  [\s(]H\.R\.   )    # Don't end sentence on "H.R."
        (?<!  ^H\.R\.    )
        (?<!  [\s(]H\.      )    # Don't end sentence on "H."
        (?<!  ^H\.       )
        (?<!  [\s(]S\.      )    # Don't end sentence on "S."
        (?<!  ^S\.       )
        (?<!  [\s(]Res\.    )    # Don't end sentence on "Res."
        (?<!  ^Res\.     )
        (?<!  [\s(]B\.C\.   )    # Don't end sentence on "B.C."
        (?<!  ^B\.C\.    )
        (?<!  [\s(]A\.D\.   )    # Don't end sentence on "A.D."
        (?<!  ^A\.D\.    )
        (?<!  [\s(]Lt\.     )    # Don't end sentence on "Lt."
        (?<!  ^Lt\.      )
        (?<!  [\s(]LT\.     )    # Don't end sentence on "LT."
        (?<!  ^LT\.      )

        (?<!  \sNo\.     )    # Don't end sentence on "No."
        (?<!  ^No\.      )

        \s+               # Split on whitespace between sentences.
        """, re.VERBOSE)

# Data on legislators, to help us match stragglers with their parties
legisc=json.load(open(indir + 'legislators-current.json', 'r'))
legish=json.load(open(indir + 'legislators-historical.json', 'r'))
legislators=legisc+legish

def get_party(id_, doc):
    congMember=[ cm for cm in doc.xpath('//congMember') if cm.get('bioGuideId')==id_ ][0]
    return congMember.get('party')

def open_mods_file(modsfn):
    xml=open(modsfn, 'r').read()
    xml=xml.replace('xmlns="http://www.loc.gov/mods/v3" ', '')
    doc=lxml.etree.fromstring(xml)
    return doc

def get_sentences(text, mode = "speech"):
    if mode == "speech":
        # We don't care about breaking the speech up into sentences, so just
        # return the whole thing
        return [text]
    return [ s for s in re.split(re_sent_end, text) if s.strip() ]

def json_to_txt(mode = "speech", to_df = False):
    for year in map(str, years):
        year_dir='{}{}/'.format(RAW_DATA_DIR, year)
        sessions=os.listdir(year_dir)
        for session in sessions:
            # No H/S proceeding for 1999-11-12
            if session=='CREC-1999-11-12':
                continue
            if session[:9]!='CREC-'+year:
                continue
            outdir='{}{}/{}/'.format(PROC_TXT_DIR, year, session)
            os.system('mkdir -p {}'.format(outdir))
            modsfn='{}{}/mods.xml'.format(year_dir, session)
            doc=open_mods_file(modsfn)
            session_dir='{}{}/json/'.format(year_dir, session)
            dspeech, rspeech, uspeech = [], [], []
            for part in os.listdir(session_dir):
                if 'PgH' not in part:
                    continue
                logging.info('Processing {}'.format(part))
                with open(session_dir+part, 'r') as seshfh:
                    json_obj=json.load(seshfh)
                    speeches=[ thing for thing in json_obj['content'] if
                               thing['kind']=='speech' ]
                for speech in speeches:
                    id_=speech['speaker_bioguide']
                    if isinstance(id_, type(None)):
                        continue
                    text=speech['text']
                    speaker=re.match(re_speaking, text).group()
                    text=text[len(speaker):]
                    # For some reason, the speaker's name is included again at
                    # the beginning of their speech. Get rid of it. This regex
                    # also sometimes removes leading parts of speech that
                    # address someone, e.g. "Madam Speaker".
                    address=re.match(re_address, text,
                                        re.IGNORECASE)
                    if not isinstance(address, type(None)):
                        text=text[len(address.group()):]
                    sents = get_sentences(text, mode = mode)
                    if mode == "unprocessed":
                        speech_ = [ sent.replace("\n", " ") for sent in sents ]
                    else:
                        tokens = map(lambda sent: tokenize(sent,
                                                           ignore_ascii = True),
                                     sents)
                        speech_ = [ ' '.join(toks) for toks in tokens ]
                    if mode == "speech":
                        assert len(speech_) == 1

                    if id_!='None':
                        party=get_party(id_, doc)
                        if party=='D':
                            dspeech.extend([ (speaker.strip(), part, s) for s in
                                             speech_ ])
                        elif party=='R':
                            rspeech.extend([ (speaker.strip(), part, s) for s in
                                             speech_ ])
                        else:
                            continue
                    else:
                        uspeech.extend([ (speaker.strip(), part, s) for s in
                                         speech_ ])
            ddf = pd.DataFrame(dspeech, columns = [ "speaker", "fn", "speech" ])
            rdf = pd.DataFrame(rspeech, columns = [ "speaker", "fn", "speech" ])
            udf = pd.DataFrame(uspeech, columns = [ "speaker", "fn", "speech" ])

            # Save
            if to_df:
                if not ddf.empty:
                    with open(outdir + "dem_" + mode, "wb") as wfh:
                        pickle.dump(ddf, wfh)
                if not rdf.empty:
                    with open(outdir + "repub_" + mode, "wb") as wfh:
                        pickle.dump(rdf, wfh)
                if not udf.empty:
                    with open(outdir + "unclassified_" + mode, "wb") as wfh:
                        pickle.dump(udf, wfh)
            else:
                if not ddf.empty:
                    with open(outdir + "dem_" + mode, "w") as wfh:
                        wfh.write("\n".join(ddf.speech))
                if not rdf.empty:
                    with open(outdir + "repub_" + mode, "w") as wfh:
                        wfh.write("\n".join(rdf.speech))
                if not udf.empty:
                    with open(outdir + "unclassified_" + mode, "w") as wfh:
                        wfh.write("\n".join(udf.speech))

def date_str_to_unix_timestamp(date_str):
    return time.mktime(time.strptime(date_str, '%Y-%m-%d'))

def get_stragglers_party(f, date):
    lname=re.search(re_lastname, f).group().lower()
    party=set()
    frule=lambda x:x['name']['last'].lower()==lname
    ls=filter(frule, legislators)
    if len(ls)==0:
        # Sometimes first names are recorded, too...
        name_=lname.split()
        fname, lname=name_[0], ' '.join(name_[1:])
        frule=lambda x:( x['name']['last'].lower()==lname and
                         x['name']['first'].lower()==fname )
        ls=filter(frule, legislators)
    for legislator in ls:
        for term in legislator['terms']:
            try:
                stdate=date_str_to_unix_timestamp(term['start'])
                enddate=date_str_to_unix_timestamp(term['end'])
            except ValueError as e:
                if e.message=='year out of range':
                    continue
                else:
                    raise e
            if stdate<=date<=enddate:
                party.add(term['party'])
    if len(party)==0:
        return 'U' # U for uncategorized
    # Make sure there is exactly one match for this person (or at least multiple
    # matches are from the same party)
    assert len(party)==1
    return list(party)[0][0]

# Not run for analyses submitted for publication.
def categorize_stragglers():
    # To record the congresspeople whose party can't be determined
    ufh=open('uncategorized', 'w')

    for year in os.listdir(PROC_TXT_DIR):
        year_dir='{}{}/'.format(PROC_TXT_DIR, year)
        for session in os.listdir(year_dir):
            session_dir='{}{}/'.format(year_dir, session)
            # Get date and convert to date object
            date=session.lstrip('CREC-')
            date=date_str_to_unix_timestamp(date)
            fs=os.listdir(session_dir)
            fs.remove('dem_speech')
            fs.remove('repub_speech')
            for f in fs:
                # Determine party affiliation
                party=get_stragglers_party(f, date)
                if party=="U":
                    ufh.write(f)
                if party not in ( "D", "R" ):
                    continue
                fout="dem_speech" if party=="D" else "repub_speech"
                cmd="cat {} >> {}".format(f, fout)
                print cmd
                # TODO: CHECK THIS BEFORE RUNNING.
                #status=subprocess.call(shlex.split(cmd))
                #assert status==0

    ufh.close()

if __name__=='__main__':
    mode = sys.argv[1].lstrip("--mode=")
    assert mode in ("sentence", "speech")

    startLog("json_to_txt-{}".format(mode))
    json_to_txt(mode)

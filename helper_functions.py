#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Author: Eran Yogev
#  Date: 14/11/2015
#
from __future__ import division

import re
import string
import pandas as pd
from nltk.corpus import stopwords
from sqlalchemy import create_engine

try:  # Python2
    from cStringIO import StringIO
except:  # Python3
    from io import StringIO

import time
import os




def run_query(query):
    connection_string = 'postgresql://postgres@localhost:5432/postgres'
    engine = create_engine(connection_string)
    return pd.read_sql_query(query, con=engine)


class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, data=[], N=None, missingfn=None, filter_level=2):
        for key, count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1. / N)
        self.stop = set(stopwords.words('english')) or set(string.punctuation)
        self.filter_level = filter_level

    def __call__(self, key):
        if key in self:
            return self[key] / self.N
        else:
            return self.missingfn(key, self.N)

    def relative_count(self, key):
        if key in self:
            return self[key]
        else:
            return 10. / (10 ** len(key))

    def relative_prob(self, fdist):
        srted = sorted(fdist.iteritems(), key=lambda tup: tup[1])

        new_N = 0
        for w, f in srted: new_N += self.relative_count(w)

        appears_more = []
        for w, f in srted:
            if ((self.relative_count(w) / new_N) * self.filter_level < fdist.freq(w)) and w not in self.stop:
                appears_more.append(w)

        return appears_more


def avoid_long_words(key, N):
    "Estimate the probability of an unknown word."
    return 10. / (N * 10 ** len(key))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def tagToValue(tag):
    tag = str(tag).replace(" ", "")
    if tag == "A1":
        return 1
    elif tag == "C2":
        return 2
    elif tag == "C1":
        return 3
    elif tag == "I":
        return 4
    elif tag == "A2":
        print("A2 FOUND")
        return 0
    else:
        raise Exception("Unknown Tag: " + str(tag))


def is_comparative(txt):
    txt = txt.lower()
    if re.search(r"as (\w+) as", txt) is not None:
        return True
    if re.search(r"is (\w+) than", txt) is not None:
        return True
    if re.search(r"are (\w+) than", txt) is not None:
        return True

    if "similar" in txt:
        return True
    if "compar" in txt:
        return True
    if "likewise" in txt:
        return True
    if "just as" in txt:
        return True
    if "same" in txt:
        return True
    if "differ" in txt:
        return True
    if "have in common" in txt:
        return True
    if "contrary" in txt:
        return True
    if "on the other hand" in txt:
        return True
    if "contrast" in txt:
        return True
    if "whereas" in txt:
        return True
    if "contrast" in txt:
        return True
    if "same as" in txt:
        return True
    if "is like" in txt:
        return True
    if "are like" in txt:
        return True
    if "relate" in txt:
        return True
    if "analog" in txt:
        return True
    if "correlat" in txt:
        return True
    if "equivalent" in txt:
        return True
    return False


def is_question(txt):
    for w in "who,where,when,why,what,which,how,?".split():
        if w in txt.lower(): return True
    return False


def is_elaboration(txt):
    for w in "because,since,cause,due to,owing to,therefore,consequently,as a result,thus,cause".split():
        if w in txt.lower(): return True
    return False


def get_credentials():
    with open("C:\\Users\\hp\\Documents\\IBMcred.txt", 'r') as f:
        return f.readlines()


def watson(txt):
    user, password = get_credentials()
    from watson_developer_cloud import ToneAnalyzerV3Beta

    succeded = False
    while not succeded:
        try:
            tone_analyzer = ToneAnalyzerV3Beta(username=user, password=password, version='2016-02-11')
            response = tone_analyzer.tone(text=txt)
            succeded = True
        except:
            time.sleep(5)
            # print(txt)
            # print(sys.exc_info())

    anger = response['document_tone']['tone_categories'][0]['tones'][0]['score']
    disgust = response['document_tone']['tone_categories'][0]['tones'][1]['score']
    fear = response['document_tone']['tone_categories'][0]['tones'][2]['score']
    joy = response['document_tone']['tone_categories'][0]['tones'][3]['score']
    sadness = response['document_tone']['tone_categories'][0]['tones'][4]['score']

    analytical = response['document_tone']['tone_categories'][1]['tones'][0]['score']
    confident = response['document_tone']['tone_categories'][1]['tones'][1]['score']
    tentative = response['document_tone']['tone_categories'][1]['tones'][2]['score']

    openess = response['document_tone']['tone_categories'][2]['tones'][0]['score']
    conscientiousness = response['document_tone']['tone_categories'][2]['tones'][1]['score']
    extraversion = response['document_tone']['tone_categories'][2]['tones'][2]['score']
    agreeableness = response['document_tone']['tone_categories'][2]['tones'][3]['score']
    neuroticism = response['document_tone']['tone_categories'][2]['tones'][4]['score']

    '''print("Anger:{0}\n"
          "disgust:{1}\n"
          "fear:{2}\n"
          "joy:{3}\n"
          "sadness:{4}\n\n"
          "analytical:{5}\n"
          "confident:{6}\n"
          "tentative:{7}\n\n"
          "openess:{8}\n"
          "conscientiousness:{9}\n"
          "extraversion:{10}\n"
          "agreeableness:{11}\n"
          "neuroticism:{12}\n".format(anger,disgust,fear,joy,sadness,
                                      analytical,confident,tentative,
                                      openess,conscientiousness,extraversion,agreeableness,neuroticism))'''

    return [anger, disgust, fear, joy, sadness, analytical, confident, tentative, openess, conscientiousness, \
            extraversion, agreeableness, neuroticism]


import pickle


class Watson_hash():
    def __init__(self):
        self.file_name = 'ce_data/watson.b'
        if os.path.isfile(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = dict()

    def get_watson(self, txt):
        if txt in self.data:
            return self.data[txt]
        else:
            tmp = watson(txt)
            self.data[txt] = tmp
            if len(self.data) % 50 == 0: print("Records watsoned: {0}".format(len(self.data)))
            with open(self.file_name, 'wb') as f:
                pickle.dump(self.data, f)
            return self.data[txt]

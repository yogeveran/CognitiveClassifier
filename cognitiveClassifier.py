# ---------------------------------------------
#               Imports
# ---------------------------------------------
import os
import pickle
import nltk
import pandas as pd
import numpy as np
from PyDictionary import PyDictionary


#Training model
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

#Feature Extraction
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Feature selection
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from textblob import TextBlob

from helper_functions import run_query, is_question, is_elaboration, is_comparative, Watson_hash


watsoner = Watson_hash()
# ---------------------------------------------
#               Load Data
# ---------------------------------------------
sheet_name = "Sheet1"
file_name = "D:\\Dropbox (BGU)\\Eran Work\\Nota-Bene\\document comment similarity\\ce_data\\all_comments.xlsx"
tagged_file = pd.read_excel(file_name, sheetname=sheet_name)
tagged_file = tagged_file[tagged_file.tag.notnull()]


# ---------------------------------------------
#               Add additional features from DB
# ---------------------------------------------
qry = 'SELECT id,(CASE  WHEN parent_id>0 THEN TRUE ELSE FALSE END) as is_comment FROM base_comment where type!=1'
tbl = run_query(qry)

merged = pd.merge(left=tagged_file,right=tbl, left_on="comment_id",right_on="id")




train, test = merged[:int(0.8*len(merged))], merged[int(0.8*len(merged)):]


# ---------------------------------------------
#               Divide to Train, Validation sets
# ---------------------------------------------
trainX = train
trainY = train.tag
testX = test
testY = test.tag
pyd = PyDictionary()

# ---------------------------------------------
#               Classes
# ---------------------------------------------
class TextFeatures(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def __init__(self):
        self.dicts_syn = dict()
        self.dicts_aton = dict()

    def get_feature_names(self):
            #return self.data[0].keys()
            return ['char_length',
                 'num_dots',
                 'num_exclamation',
                 'num_question_mark',
                 'num_words',
                 'polarity',
                 'subjectivity',
                 'comparison_syn_count',
                 'comparison_atonym_count',
                 'like_syn_count',
                 'like_atonym_count',
                 'confused_syn_count',
                 'confused_atonym_count',
                 'understand_syn_count',
                 'understand_atonym_count',
                 'idea_syn_count',
                 'idea_atonym_count',

                 'agree_syn_count',
                 'agree_atonym_count',
                 'thanks_syn_count',
                 'thanks_atonym_count',

                 'interesting_syn_count',
                 'interesting_atonym_count',

                 'amazing_syn_count',
                 'amazing_atonym_count',

                 'is_comparative',
                 'is_question',
                 'is_elaboration',

                 #watson features
                 'anger',
                 'disgust',
                 'fear',
                 'joy',
                 'sadness',
                 'analytical',
                 'confident',
                 'tentative',
                 'openess',
                 'conscientiousness',
                 'extraversion',
                 'agreeableness',
                 'neuroticism']

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        self.data = [{'char_length': len(text),
                 'num_dots': text.count('.'),
                 'num_exclamation': text.count('!'),
                 'num_question_mark': text.count('?'),
                 'num_words': len(nltk.word_tokenize(text)),
                 'polarity': TextBlob(text).sentiment.polarity,
                 'subjectivity': TextBlob(text).sentiment.subjectivity,
                 'comparison_syn_count': self.count_syn_in_text(text,'comparison'),
                 'comparison_atonym_count': self.count_aton_in_text(text, 'comparison'),
                 'like_syn_count': self.count_syn_in_text(text, 'like'),
                 'like_atonym_count': self.count_aton_in_text(text, 'like'),
                 'confused_syn_count': self.count_syn_in_text(text, 'confused'),
                 'confused_atonym_count': self.count_aton_in_text(text, 'confused'),
                 'understand_syn_count': self.count_syn_in_text(text, 'understand'),
                 'understand_atonym_count': self.count_aton_in_text(text, 'understand'),
                 'idea_syn_count': self.count_syn_in_text(text, 'idea'),
                 'idea_atonym_count': self.count_aton_in_text(text, 'idea'),

                 'agree_syn_count': self.count_syn_in_text(text, 'agree'),
                 'agree_atonym_count': self.count_aton_in_text(text, 'agree'),
                 'thanks_syn_count': self.count_syn_in_text(text, 'thanks'),
                 'thanks_atonym_count': self.count_aton_in_text(text, 'thanks'),

                 'interesting_syn_count': self.count_syn_in_text(text, 'interesting'),
                 'interesting_atonym_count': self.count_aton_in_text(text, 'interesting'),

                 'amazing_syn_count': self.count_syn_in_text(text, 'amazing'),
                 'amazing_atonym_count': self.count_aton_in_text(text, 'amazing'),

                 'is_comparative': is_comparative(text),
                 'is_question': is_question(text),
                 'is_elaboration': is_elaboration(text),

                 #watson features
                 'anger': watsoner.get_watson(text)[0],
                 'disgust': watsoner.get_watson(text)[1],
                 'fear': watsoner.get_watson(text)[2],
                 'joy': watsoner.get_watson(text)[3],
                 'sadness': watsoner.get_watson(text)[4],
                 'analytical': watsoner.get_watson(text)[5],
                 'confident': watsoner.get_watson(text)[6],
                 'tentative': watsoner.get_watson(text)[7],
                 'openess': watsoner.get_watson(text)[8],
                 'conscientiousness': watsoner.get_watson(text)[9],
                 'extraversion': watsoner.get_watson(text)[10],
                 'agreeableness': watsoner.get_watson(text)[11],
                 'neuroticism': watsoner.get_watson(text)[12],

                 } for text in posts]
        return self.data

    def count_syn_in_text(self, text, syn):
        if syn in self.dicts_syn:
            syns = self.dicts_syn[syn]
        else:
            syns = self.create_if_not_exist('dictionaries/{0}_syn.b'.format(syn),lambda : pyd.synonym(syn))
            self.dicts_syn[syn] = syns
        return len([1 for w in syns if w in text.lower()])

    def count_aton_in_text(self, text, aton):
        if aton in self.dicts_aton:
            atons = self.dicts_aton[aton]
        else:
            atons = self.create_if_not_exist('dictionaries/{0}_aton.b'.format(aton), lambda : pyd.synonym(aton))
            self.dicts_aton[aton] = atons
        return len([1 for w in atons if w in text.lower()])

    def create_if_not_exist(self,fileName,constructor):
        if os.path.isfile(fileName):
            with open(fileName, 'rb') as f:
                return pickle.load(f)
        else:
            obj = constructor()
            with open(fileName, 'wb') as f:
                pickle.dump(obj,f)
            return obj

class MetaFeatures(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'is_comment': line.is_comment,
                 #'num_dots': line.body.count('.'),
                 } for index, line in posts.iterrows()]

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class DenseTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None):
        return X.todense()

    def fit(self, X, y=None, **fit_params):
        return self
# ---------------------------------------------
#               Classify
# ---------------------------------------------

pipe = Pipeline([
    ("DataPreparation", FeatureUnion(
        [

            ('MetaFeatures_pipe', Pipeline([
                ('MetaFeatures', MetaFeatures()),  # returns a list of dicts
                ('MetaFeatures_vect', DictVectorizer(sparse=False)),  # list of dicts -> feature matrix
            ])),

            ('bow_features_pipe', Pipeline([
                ('bow_features', ItemSelector(key='body')),
                ("bow_features_vect",
                 CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=100,
                                 ngram_range=(1, 2))),
                ('bow_features_dense', DenseTransformer()),
            ])),

            ('TextFeatures_pipe', Pipeline([
                ('TextFeatures_selector', ItemSelector(key='body')),
                ('TextFeatures', TextFeatures()),  # returns a list of dicts
                ('TextFeatures_vect', DictVectorizer(sparse=False)),  # list of dicts -> feature matrix
            ]))
        ]
    )),

    #('classifier', AdaBoostClassifier(n_estimators=10))  # f1 - 0.55 finds all labels
    #('classifier',SVC(degree=10, tol=0.0001)) #f1 - 0.5 finds all labels
    #('classifier', RFECV(LDA(solver='eigen', shrinkage='auto', priors=None, n_components=None, store_covariance=False, tol=0.001), step=1, cv=3)) # 0.54 finds mostly c2 and i1
    #('classifier', RFECV(tree.DecisionTreeClassifier(), step=1, cv=3))# 0.58 - finds all labels
    ('classifier', tree.DecisionTreeClassifier())# 0.60 - finds all labels
    #('classifier', tree.DecisionTreeClassifier(criterion='entropy'))  # 0.58 - finds all labels
    ])
pipe.fit(trainX,trainY)
predicted = pipe.predict(testX)
# ---------------------------------------------
#               Show Results
# ---------------------------------------------
print(confusion_matrix(testY,predicted,labels=['A1','C2','C1','I']))
print(classification_report(testY,predicted,labels=['A1','C2','C1','I']))

def paint_tree():
    import pydot_ng as pydot
    from sklearn.externals.six import StringIO
    params = pipe.get_params()
    #for key, value in params.items(): print(key,value)
    feats = params['DataPreparation__MetaFeatures_pipe__MetaFeatures_vect'].get_feature_names()
    feats += params['DataPreparation__bow_features_pipe__bow_features_vect'].get_feature_names()
    feats += params['DataPreparation__TextFeatures_pipe__TextFeatures_vect'].get_feature_names()
    dot_data = StringIO()
    clf = pipe.get_params()['classifier']
    tree.export_graphviz(clf, out_file=dot_data,
                         class_names=['A1', 'C2', 'C1', 'I'],
                         feature_names=feats,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("DecisionTree.pdf")


paint_tree()



from utils.csv_loader import load_lightship_data
from utils.unhash_data import load_hash_tables
from pipelines import build_pipeline_bagofwords
from pipelines import build_pipeline_word2vec
from pipelines import build_pipeline_A

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

import gensim.downloader as api
from gensim.models import KeyedVectors

import numpy as np
import os

files = ['task.csv',
         'task_title_keyword_hashes.csv',
         'task_details_keyword_hashes.csv']

ls = load_lightship_data(files)

hash_table = load_hash_tables([ls['task_title_keyword_hashes'],
                               ls['task_details_keyword_hashes']])

task_df = ls['task']
task_df = task_df[task_df['assignee_id'].notnull()]

kv_list = None

w2v_modelname = 'glove-50.kv'
if os.path.exists(w2v_modelname):
    kv_list = KeyedVectors.load(w2v_modelname)
else:
    kv_list = api.load('glove-wiki-gigaword-50')
    kv_list.save(w2v_modelname)

bagofwords_pipe = make_pipeline(
    build_pipeline_bagofwords(hash_table),
    LogisticRegression()
)

word2vec_pipe = make_pipeline(
    build_pipeline_word2vec(hash_table, kv_list),
    GradientBoostingClassifier()
)

task_pipe = make_pipeline(
    build_pipeline_A(),
    GradientBoostingClassifier()
)

clf = StackingClassifier([('bow', bagofwords_pipe),
                          ('w2v', word2vec_pipe),
                          ('task', task_pipe)])

y = task_df['assignee_id']

scores = cross_val_score(clf, task_df, y, cv=4, scoring='accuracy')
print(scores.mean())

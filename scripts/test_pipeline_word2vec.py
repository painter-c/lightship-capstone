from utils.csv_loader import load_lightship_data
from utils.unhash_data import load_hash_tables
from pipelines import build_pipeline_word2vec

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import gensim.downloader as api
from gensim.models import KeyedVectors

import os

files = ['task.csv',
         'task_title_keyword_hashes.csv',
         'task_details_keyword_hashes.csv']

ls = load_lightship_data(files)

hash_table = load_hash_tables([ls['task_title_keyword_hashes'],
                               ls['task_details_keyword_hashes']])

task_df = ls['task']
task_df = task_df[task_df['assignee_id'].notnull()]

kw_df = task_df[['title', 'details']]

kv_list = None

w2v_modelname = 'glove-50.kv'
if os.path.exists(w2v_modelname):
    kv_list = KeyedVectors.load(w2v_modelname)
else:
    kv_list = api.load('glove-wiki-gigaword-50')
    kv_list.save(w2v_modelname)

clf_pipe = make_pipeline(
    build_pipeline_word2vec(hash_table, kv_list),
    GradientBoostingClassifier()
)

y = task_df['assignee_id']

scores = cross_val_score(clf_pipe, kw_df, y, scoring='balanced_accuracy')
print(scores.mean())
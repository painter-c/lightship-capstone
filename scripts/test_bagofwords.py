from utils.csv_loader import load_lightship_data
from utils.unhash_data import load_hash_tables
from pipelines import build_pipeline_bagofwords

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np


files = ['task.csv',
         'task_title_keyword_hashes.csv',
         'task_details_keyword_hashes.csv']

ls = load_lightship_data(files)

hash_table = load_hash_tables([ls['task_title_keyword_hashes'],
                               ls['task_details_keyword_hashes']])

task_df = ls['task']
task_df = task_df[task_df['assignee_id'].notnull()]

kw_df = task_df[['title', 'details']]

clf_pipe = make_pipeline(
    build_pipeline_bagofwords(hash_table),
    LogisticRegression()
    #GradientBoostingClassifier() <- doesn't work as well
)

y = task_df['assignee_id']

scores = cross_val_score(clf_pipe, kw_df, y, scoring='accuracy')
print(scores.mean())
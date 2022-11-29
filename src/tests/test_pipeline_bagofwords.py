import src.utils.config as config

from src.utils.csv_loader import load_lightship_data
from src.utils.unhash_data import load_hash_tables
from src.pipelines import build_pipeline_bagofwords

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

cfg = config.load()

files = ['task.csv',
         'task_title_keyword_hashes.csv',
         'task_details_keyword_hashes.csv']

ls = load_lightship_data(cfg, files)

ht = load_hash_tables([ls['task_title_keyword_hashes'],
                       ls['task_details_keyword_hashes']])

task_df = ls['task']
task_df = task_df[task_df['assignee_id'].notnull()]
task_df = task_df[task_df['creator_id'].ne('4b5f8672-2180-4507-a694-4926e0da7f83')]

kw_df = task_df[['title', 'details']]

clf_pipe = make_pipeline(
    build_pipeline_bagofwords(ht),
    LogisticRegression(max_iter=1000)
)

y = task_df['assignee_id']

scores = cross_val_score(clf_pipe, kw_df, y, cv=3, scoring='accuracy')
print(scores.mean())
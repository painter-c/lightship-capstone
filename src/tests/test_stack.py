import src.utils.config as config
import src.utils.util_misc as util_misc

from src.utils.csv_loader import load_lightship_data
from src.utils.unhash_data import load_hash_tables
from src.pipelines import build_pipeline_bagofwords
from src.pipelines import build_pipeline_word2vec
from src.pipelines import build_pipeline_A

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import top_k_accuracy_score

import gensim.downloader as api
from gensim.models import KeyedVectors

import os

cfg = config.load()

ls = load_lightship_data(cfg['datasets']['set_1'],
                         ['task.csv',
                          'task_title_keyword_hashes.csv',
                          'task_details_keyword_hashes.csv',
                          'account.csv'])

ht = load_hash_tables([ls['task_title_keyword_hashes'],
                       ls['task_details_keyword_hashes']])

task_df = ls['task']
task_df = task_df[task_df['assignee_id'].notnull()]

model_name = 'glove-wiki-gigaword-50'
model_path = cfg['cache_location'] + model_name + '.kv'

kv_list = None
if os.path.exists(model_path):
    kv_list = KeyedVectors.load(model_path)
else:
    kv_list = api.load(model_name)
    kv_list.save(model_path)

bagofwords_pipe = make_pipeline(
    build_pipeline_bagofwords(ht),
    LogisticRegression()
)

word2vec_pipe = make_pipeline(
    build_pipeline_word2vec(ht, kv_list),
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

# scores = cross_val_score(clf, task_df, y, cv=4, scoring='accuracy')
# print(scores.mean())

# test mapping probabilities to names
acc_dict = util_misc.get_account_name_dict(ls['account'])

X_train, X_test, y_train, y_test = train_test_split(task_df, y, test_size = 0.7)

clf.fit(X_train, y_train)

ps = clf.predict_proba(X_test)

for i in range(ps.shape[0]):
    p_row = ps[i]
    print(f'Expected: {acc_dict[y_test.iat[i]]}')
    reccs = util_misc.get_acc_reccomendations(p_row,
                                              clf.classes_,
                                              acc_dict)
    for j, recc in enumerate(reccs):
        print(f'{j}. {recc.name} {recc.account_id} {recc.probability:.10f}')
    print()
    
y_pred = clf.predict(X_test)
score = top_k_accuracy_score(y_test, ps, k=3, labels=clf.classes_)
print(score)
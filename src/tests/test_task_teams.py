# Classification of task teams

import src.utils.config as config_manager
import src.utils.common as common
import src.pipelines as pipelines
import src.transformers as tfm
from src.transforms import team_target_transform
from src.utils.loading import LightshipLoader

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

config = config_manager.load()
loader = LightshipLoader(config)

assignee_blacklist = config_manager.load_assignee_blacklist()
creator_blacklist = config_manager.load_creator_blacklist()

task_df = loader.load('task.csv')
task_teams = loader.load('task_teams.csv')

keyword_table = loader.load_keyword_table()

# Build a dataset where examples with multiple team observers are duplicated, 
# one team observer per example.
task_team_dict = common.get_task_team_dict(task_teams)
task_df = team_target_transform(task_df, task_team_dict)

text_columns = ['title', 'details']

unhash_transformer = None
if config['unhashing_enabled']:
    unhash_transformer = tfm.HashDecoder(loader.load_keyword_table(), text_columns)

stem_transformer = None
if config['stemming_enabled']:
    stem_transformer = tfm.WordStemmer(text_columns)

preprocess = make_pipeline(
    tfm.BlacklistFilter(creator_blacklist, 'creator_id'),
    unhash_transformer,
    tfm.WordTokenizer(text_columns),
    tfm.StopwordFilter(text_columns),
    stem_transformer,
    tfm.WordTokenJoin(text_columns)
)

task_df = preprocess.fit_transform(task_df)

X = task_df.drop('team_id', axis=1)
y = task_df['team_id']

clf = StackingClassifier([
    ('tsk', pipelines.pipeline_task_teams()),
    ('bow', pipelines.pipeline_count_vectorizer())
])

scores = cross_val_score(clf, X, y, scoring='accuracy', cv=3)
print('accuracy', scores.mean())

scores = cross_val_score(clf, X, y, scoring='roc_auc_ovr', cv=3)
print('roc auc ovr', scores.mean())
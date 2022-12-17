import src.utils.config as config_manager
import src.transformers as tfm
import src.pipelines as pipelines
from src.utils.loading import LightshipLoader

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

import gensim.downloader as gensim_api
import gensim.models as gensim_models

config = config_manager.load()
loader = LightshipLoader(config)

assignee_blacklist = config_manager.load_assignee_blacklist()
creator_blacklist = config_manager.load_creator_blacklist()

task_df = loader.load('task.csv')

model = 'glove-wiki-gigaword-50'

kv_model = gensim_models.KeyedVectors.load(
    config['cache_location'] + model + '.kv')

if kv_model is None:
    kv_model = gensim_api.load(model)
    kv_model.save(config['cache_location'] + model + '.kv')

columns = ['title', 'details']

unhash_transformer = None
if config['unhashing_enabled']:
    unhash_transformer = tfm.HashDecoder(loader.load_keyword_table(), columns)

preprocess = make_pipeline(
    tfm.NullEntryFilter('assignee_id'),
    tfm.BlacklistFilter(creator_blacklist, 'creator_id'),
    tfm.BlacklistFilter(assignee_blacklist, 'assignee_id'),
    tfm.LowFrequencyFilter('assignee_id', config['min_class_frequency']),
    unhash_transformer,
    tfm.WordTokenizer(columns),
    tfm.StopwordFilter(columns),
    tfm.WordTokenJoin(columns)
)

task_df = preprocess.fit_transform(task_df)

X = task_df[['title', 'details']]
y = task_df['assignee_id']

scores = cross_val_score(
    pipelines.pipeline_word_vectorizer(kv_model), X, y,
    cv=3, scoring='roc_auc_ovr')

print('roc auc ovr:', scores.mean())

scores = cross_val_score(
    pipelines.pipeline_word_vectorizer(kv_model), X, y,
    cv=3, scoring='accuracy')

print('accuracy:', scores.mean())
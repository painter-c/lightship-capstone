import src.utils.config as config_manager
import src.transformers as tfm
import src.pipelines as pipelines
from src.utils.loading import LightshipLoader

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

config = config_manager.load()
loader = LightshipLoader(config)

assignee_blacklist = config_manager.load_assignee_blacklist()
creator_blacklist = config_manager.load_creator_blacklist()

task_df = loader.load('task.csv')

columns = ['title', 'details']

unhash_transformer = None
if config['unhashing_enabled']:
    unhash_transformer = tfm.HashDecoder(loader.load_keyword_table(), columns)

stem_transformer = None
if config['stemming_enabled']:
    stem_transformer = tfm.WordStemmer(columns)

preprocess = make_pipeline(
    tfm.NullEntryFilter('assignee_id'),
    tfm.BlacklistFilter(creator_blacklist, 'creator_id'),
    tfm.BlacklistFilter(assignee_blacklist, 'assignee_id'),
    tfm.LowFrequencyFilter('assignee_id', config['min_class_frequency']),
    unhash_transformer,
    tfm.WordTokenizer(columns),
    tfm.StopwordFilter(columns),
    stem_transformer,
    tfm.WordTokenJoin(columns)
)

task_df = preprocess.fit_transform(task_df)

X = task_df[['title', 'details']]
y = task_df['assignee_id']

scores = cross_val_score(
    pipelines.pipeline_count_vectorizer(), X, y,
    cv=3, scoring='roc_auc_ovr')

print('roc auc ovr:', scores.mean())

scores = cross_val_score(
    pipelines.pipeline_count_vectorizer(), X, y,
    cv=3, scoring='accuracy')

print('accuracy:', scores.mean())
import src.utils.config as config_manager
import src.transformers as tfm
from src.utils.loading import LightshipLoader

from sklearn.pipeline import make_pipeline
import pandas as pd

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
    #tfm.BlacklistFilter(creator_blacklist, 'creator_id'),
    #tfm.BlacklistFilter(assignee_blacklist, 'assignee_id'),
    #tfm.LowFrequencyFilter('assignee_id', config['min_class_frequency']),
    unhash_transformer,
    tfm.WordTokenizer(columns),
    tfm.StopwordFilter(columns),
    stem_transformer,
    tfm.WordTokenJoin(columns)
)

task_df = preprocess.fit_transform(task_df)

task_df['text'] = task_df['title'] + ' ' + task_df['details']
task_df['text'].fillna('empty', inplace=True)

corpus = task_df['text'].to_numpy()

from bertopic import BERTopic
from umap import UMAP

umap_model = UMAP(n_neighbors=15,
                  n_components=5,
                  min_dist=0.0,
                  metric='cosine',
                  random_state=100)

topic_model = BERTopic(umap_model=umap_model,
                       language='english',
                       calculate_probabilities=True)

topics, probabilities = topic_model.fit_transform(corpus)

task_df['topics'] = pd.Series(topics)

print(topic_model.get_topic_info())
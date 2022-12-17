from src.transformers import BlacklistFilter, NullEntryFilter, StopwordFilter
from src.transformers import HashDecoder, LowFrequencyFilter, WordTokenizer
from src.utils.loading import LightshipLoader
import src.utils.config as config_manager
from sklearn.pipeline import make_pipeline


config = config_manager.load()
loader = LightshipLoader(config)

task_df = loader.load('task.csv')
keyword_table = loader.load_keyword_table()


def test_blacklisted_account_filter():
    
    blacklist = ['4b5f8672-2180-4507-a694-4926e0da7f83']
    
    blacklist_filter = BlacklistFilter(blacklist, 'creator_id')
    
    return blacklist_filter.fit_transform(task_df)
    
blacklist_df = test_blacklisted_account_filter()

def test_null_entry_filter():
    
    null_entry_filter = NullEntryFilter('assignee_id')
    
    return null_entry_filter.fit_transform(task_df)
    
nonull_df = test_null_entry_filter()

def test_hash_decoder():
    
    hash_decoder = HashDecoder(keyword_table, ['title', 'details'])
    
    return hash_decoder.fit_transform(task_df)
    
decoded_df = test_hash_decoder()

def test_preprocessing_pipeline():
    blacklist = config_manager.load_creator_blacklist()
    transformer_list = []
    transformer_list.append(NullEntryFilter('assignee_id'))
    transformer_list.append(BlacklistFilter(blacklist, 'creator_id'))
    transformer_list.append(LowFrequencyFilter('assignee_id', 5))
    transformer_list.append(HashDecoder(keyword_table, ['title', 'details']))
    preprocessing = make_pipeline(*transformer_list)
    return preprocessing.fit_transform(task_df)

pl_df = test_preprocessing_pipeline()
    
def test_word_tokenizer():
    
    word_tokenizer = WordTokenizer(['title', 'details'])
    
    return word_tokenizer.fit_transform(decoded_df)

token_df = test_word_tokenizer()

def test_stopword_filter():
    #stopwords = loader.load_stopwords()
    stopwords = {'complete', 'review', 'create'}
    
    stopword_filter = StopwordFilter(['title', 'details'], stopwords)
    
    return stopword_filter.fit_transform(token_df)

no_stopwords = test_stopword_filter()

















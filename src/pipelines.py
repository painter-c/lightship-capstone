import src.transforms as tf

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

def clf_default():
    return LogisticRegression()

def ord_passthrough():
    return ColumnTransformer([
        ('passthrough', 'passthrough', ['project_id', 'creator_id'])
    ])

def pipeline_ohe():
    return Pipeline([
        ('ordinal-passthrough', ord_passthrough()),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

def text_passthrough():
    return ColumnTransformer([
        ('passthrough', 'passthrough', ['title', 'creator_id'])
    ])

def pipeline_count_vectorizer():
    return Pipeline([
        ('passthrough', text_passthrough()),
        ('merge-string', FunctionTransformer(tf.merge_string_cols)),
        ('count-vectorizer', CountVectorizer()),
        ('clf', clf_default())
    ])

def pipeline_word_vectorizer(kv_list):
    return Pipeline([
        ('passthrough', text_passthrough()),
        ('lowercase', FunctionTransformer(tf.to_lowercase)),
        ('duplicates', FunctionTransformer(tf.remove_duplicate_words)),
        ('vectorizer', FunctionTransformer(tf.vectorize_words, kw_args={'kv_list':kv_list})),
        ('clf', clf_default())
    ])

def pipeline_task(clf=clf_default()):
    return Pipeline([
        ('ohe', pipeline_ohe()),
        ('impute', SimpleImputer(strategy='mean')),
        ('clf', clf)
    ])

def pipeline_master(kv_list):
    return StackingClassifier([
        ('task', pipeline_task()),
        ('count', pipeline_count_vectorizer()),
        ('word', pipeline_word_vectorizer(kv_list))
    ])

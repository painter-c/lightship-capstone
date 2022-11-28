import src.transforms as T

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
        ('pass-col', 'passthrough', ['project_id', 'creator_id'])
    ])

def pipeline_ohe():
    return Pipeline([
        ('pass-ord', ord_passthrough()),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

def text_passthrough():
    return ColumnTransformer([
        ('pass-col', 'passthrough', ['title', 'creator_id'])
    ])

def pipeline_count_vectorizer():
    return Pipeline([
        ('pass-text', text_passthrough()),
        ('ft-merge', FunctionTransformer(T.merge_string_cols_transform)),
        ('count-vec', CountVectorizer()),
        ('clf', clf_default())
    ])

def pipeline_word_vectorizer(kv_list):
    return Pipeline([
        ('pass-text', text_passthrough()),
        ('ft-lower', FunctionTransformer(T.lowercase_transform)),
        ('ft-remdup', FunctionTransformer(T.remove_duplicate_words_transform)),
        ('ft-embed', FunctionTransformer(T.word_embed_transform, kw_args={'kv_list':kv_list})),
        ('clf', clf_default())
    ])

def pipeline_task():
    return Pipeline([
        ('pipe-ohe', pipeline_ohe()),
        ('impute', SimpleImputer(strategy='mean')),
        ('clf', clf_default())
    ])

def pipeline_master(kv_list):
    return StackingClassifier([
        ('pipe-task', pipeline_task()),
        ('pipe-count', pipeline_count_vectorizer()),
        ('pipe-word', pipeline_word_vectorizer(kv_list))
    ])

import src.transforms as tf

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def clf_default_task():
    return LogisticRegression(max_iter=2000, class_weight='balanced')

def clf_default_text():
    return RandomForestClassifier(class_weight='balanced')

def ord_passthrough():
    return ColumnTransformer([
        ('passthrough', 'passthrough', ['project_id', 'creator_id'])
    ])

def ord_passthrough_teams():
    return ColumnTransformer([
        ('passthrough', 'passthrough', ['project_id', 'creator_id', 'assignee_id'])
    ])

def pipeline_ohe():
    return Pipeline([
        ('ordinal-passthrough', ord_passthrough()),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

def pipeline_ohe_teams():
    return Pipeline([
        ('ordinal-passthrough', ord_passthrough_teams()),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

def text_passthrough():
    return ColumnTransformer([
        ('passthrough', 'passthrough', ['details'])
    ])

def pipeline_count_vectorizer(clf=clf_default_text()):
    return Pipeline([
        ('passthrough', text_passthrough()),
        ('flatten', FunctionTransformer(tf.flatten_column)),
        ('count-vectorizer', TfidfVectorizer()),
        ('clf', clf)
    ])

def pipeline_word_vectorizer(kv_list, clf=clf_default_text()):
    return Pipeline([
        ('passthrough', text_passthrough()),
        ('vectorizer', FunctionTransformer(tf.vectorize_words, kw_args={'kv_list':kv_list})),
        ('clf', clf)
    ])

def pipeline_task(clf=clf_default_task()):
    return Pipeline([
        ('ohe', pipeline_ohe()),
        ('impute', SimpleImputer(strategy='mean')),
        ('clf', clf)
    ])

def pipeline_task_teams(clf=clf_default_task()):
    return Pipeline([
        ('ohe', pipeline_ohe_teams()),
        ('impute', SimpleImputer(strategy='mean')),
        ('clf', clf)
    ])
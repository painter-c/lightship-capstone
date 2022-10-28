import transforms as T

from sklearn.pipeline import make_union, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer

def _build_ct_unhash(hash_table):
    return make_column_transformer(
        (FunctionTransformer(
            T.unhash_transform,
            kw_args={'hash_table':hash_table}),
         ['title', 'details'])
    )

def _build_pipeline_ohe():
    return make_pipeline(
        make_column_transformer(
            (FunctionTransformer(T.dayofweek_transform), ['created']),
            (MissingIndicator(), ['due_date']),
            ('passthrough', ['project_id', 'creator_id'])
        ),
        OneHotEncoder(handle_unknown='ignore')
    )

# Input:
#   Hashed title and details columns
#
# Output:
#   Sparse matrix of keyword occurences
#
def build_pipeline_bagofwords(hash_table):
    return make_pipeline(
        # Unhash title and details columns
        _build_ct_unhash(hash_table),
        # Combine title and details columns
        FunctionTransformer(T.combine_title_details_transform),
        # Generate sparse matrix of keyword occurences
        CountVectorizer()
    )

# Input:
#   Hashed title and detail columns
#
# Output:
#   Title and detail columns transformed into sentence-level word
#   embeddings.
#
def build_pipeline_word2vec(hash_table, kv_list):
    return make_pipeline(
        # Unhash title and details columns
        _build_ct_unhash(hash_table),
        # Generate word embeddings
        FunctionTransformer(T.word_embed_transform,
                            kw_args={'kv_list':kv_list})
    )

# Input:
#   Entire task column with assignee_id removed
#
# Output:
#   Following columns one-hot encoded:
#      Day of the week (from created)
#      Has due date (from missing values of due_date)
#      project_id
#      creator_id
#
#   A timeofday column (from created)
#   priority column
#
#   * All columns are imputed at the end of the pipeline
#
def build_pipeline_A():
    return make_pipeline(
        make_union(
            _build_pipeline_ohe(),
            make_column_transformer(
                (FunctionTransformer(T.timeofday_transform), ['created']),
                ('passthrough', ['priority'])
            )
        ),
        SimpleImputer(strategy='mean')
    )
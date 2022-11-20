from src.utils import unhash_data
from src.utils import text_misc

import numpy as np

def lowercase_transform(X):
    return np.char.lower(X.astype(str))

_rem_dup_func = np.frompyfunc(text_misc.remove_duplicates, 1, 1)

def remove_duplicate_words_transform(X):
    out = None
    for col in X.T:
        new_col = _rem_dup_func(col)
        new_col = new_col.reshape(-1, 1)
        if out is None:
            out = new_col
        else:
            out = np.hstack((out, new_col))
    return out

_extract_func = np.vectorize(text_misc.extract_words_by_tag,
                             excluded={'tag', 'spacy_nlp'})

def word_tag_transform(X, tag, spacy_nlp):
    out = None
    for col in X.T:
        new_col = _extract_func(col, tag, spacy_nlp)
        new_col = new_col.reshape(-1, 1)
        if out is None:
            out = new_col
        else:
            out = np.hstack((out, new_col))
    return out
    

def unhash_transform(df, hash_table):
    out = []
    for name in df.columns:
        col = df[name]
        col = col.apply(unhash_data.unhash_str, args=(hash_table,))
        col = col.to_numpy().reshape(-1, 1)
        if len(out):
            out = np.hstack((out, col))
        else:
            out = col
    return out

def unhash_column(column, keyword_hash):
    return column.apply(unhash_data.unhash_str, args=(keyword_hash,))

def _word_embed_transform_col(col, kv_list):
    out = None
    for text in col:
        keys = text.split() if text is not None else []
        if len(keys):
            mean_vec = kv_list.get_mean_vector(keys)
            if out is None:
                out = mean_vec
            else:
                out = np.vstack((out, mean_vec))
        else:
            #val = float('nan')
            val = 0.0
            if out is None:
                out = np.full((kv_list.vector_size,), val)
            else:
                out = np.vstack((out, np.full((kv_list.vector_size,), val)))
    return out

def word_embed_transform(X, kv_list):
    out = None
    for i_col in range(X.shape[1]):
        col = X[:,i_col]
        embed_col = _word_embed_transform_col(col, kv_list)
        if out is None:
            out = embed_col
        else:
            out = np.hstack((out, embed_col))
    return out

_merge_str_func = np.frompyfunc(text_misc.join_string, 2, 1)

def merge_string_cols_transform(X):
    X_T = X.T
    out = X_T[0]
    for i in range(1, X_T.shape[0]):
        curr = X_T[i]
        out = _merge_str_func(out, curr)
    return out

def dayofweek_transform(df):
    out = []
    for name in df.columns:
        col = df[name]
        if not str(col.dtype).count('datetime64'):
            continue
        new_col = col.dt.dayofweek
        new_col = new_col.to_numpy()
        new_col = new_col.reshape(-1, 1)
        if len(out):
            out = np.hstack((out, new_col))
        else:
            out = new_col
    return out
            
def timeofday_transform(df):
    out = []
    for name in df.columns:
        col = df[name]
        if not str(col.dtype).count('datetime64'):
            continue
        new_col = col.dt.hour
        new_col = new_col.to_numpy()
        new_col = new_col.reshape(-1, 1)
        if len(out):
            out = np.hstack((out, new_col))
        else:
            out = new_col
    return out

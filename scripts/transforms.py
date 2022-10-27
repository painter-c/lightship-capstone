import numpy as np

from utils import unhash_data

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

def _mean_word_embedding(text, kv_list):
    words = text.split()
    return kv_list.get_mean_vector(words)

def word_embedding_transform(X, kv_list):
    pass

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

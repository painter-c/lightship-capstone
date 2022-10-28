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

# Input:
#   The unhashed title and details columns
#
# Output:
#   Combined title and details columns
#
def combine_title_details_transform(X):
    # Add space to titles otherwise last word from title and first word of 
    # details will be combined.
    c_title = X[:,0] + ' '
    c_details = X[:,1]
    return np.char.add(c_title.astype(str), c_details.astype(str))

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


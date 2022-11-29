import src.utils.common as com
import numpy as np
import pandas as pd

def to_lowercase(X):
    return np.char.lower(X.astype(str))


__rem_dup_func = np.frompyfunc(com.remove_duplicates, 1, 1)

def remove_duplicate_words(X):
    out = None
    for col in X.T:
        new_col = __rem_dup_func(col)
        new_col = new_col.reshape(-1, 1)
        if out is None:
            out = new_col
        else:
            out = np.hstack((out, new_col))
    return out


_extract_func = np.vectorize(com.extract_words_by_tag,
                             excluded={'tag', 'spacy_nlp'})

def tag_words(X, tag, spacy_nlp):
    out = None
    for col in X.T:
        new_col = _extract_func(col, tag, spacy_nlp)
        new_col = new_col.reshape(-1, 1)
        if out is None:
            out = new_col
        else:
            out = np.hstack((out, new_col))
    return out
    

def unhash_keywords(df, hash_table):
    out = []
    for name in df.columns:
        col = df[name]
        col = col.apply(com.unhash_str, args=(hash_table,))
        col = col.to_numpy().reshape(-1, 1)
        if len(out):
            out = np.hstack((out, col))
        else:
            out = col
    return out


def unhash_column(column, keyword_hash):
    return column.apply(com.unhash_str, args=(keyword_hash,))


def __vectorize_word_column(col, kv_list):
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


def vectorize_words(X, kv_list):
    out = None
    for i_col in range(X.shape[1]):
        col = X[:,i_col]
        embed_col = __vectorize_word_column(col, kv_list)
        if out is None:
            out = embed_col
        else:
            out = np.hstack((out, embed_col))
    return out


__merge_str_func = np.frompyfunc(com.join_string, 2, 1)

def merge_string_cols(X):
    X_T = X.T
    out = X_T[0]
    for i in range(1, X_T.shape[0]):
        curr = X_T[i]
        out = __merge_str_func(out, curr)
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


# Each task example may have zero or more teams assigned to it as 
# collaborators. This function generates a task example for each unique pair 
# of (task_id, team_id).
def team_target_transform(task_df, team_dict):
    # Convert team dict to a list
    team_list = np.empty((0,2))
    for task_id in team_dict:
        for team_id in team_dict[task_id]:
            team_list = np.vstack((team_list, np.array((task_id, team_id))))
    team_df = pd.DataFrame(data=team_list, columns = ['task_id', 'team_id'])
    result = task_df.merge(team_df, left_on='id', right_on='task_id')
    result.drop('task_id', axis=1, inplace=True)
    return result  
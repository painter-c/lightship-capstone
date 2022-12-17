import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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


def flatten_column(column):
    return column.flatten()


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


def extract_tfidf_keywords(series, n_keywords):
    docs = series.to_numpy()
    cv = CountVectorizer(max_df=0.85)
    tf = TfidfTransformer(smooth_idf = True, use_idf = True)
    sparse_tfidf = tf.fit_transform(cv.fit_transform(docs))
    features = cv.get_feature_names()
    #
    result = []
    for index in range(1, sparse_tfidf.indptr.shape[0]):
        a = sparse_tfidf.indptr[index-1]
        b = sparse_tfidf.indptr[index]
        row_terms = [features[i] for i in sparse_tfidf.indices[a:b]]
        row_values = sparse_tfidf.data[a:b]
        #
        row = list(zip(row_terms, row_values))
        row.sort(key=lambda x: x[1], reverse=True)
        row = [item[0] for item in row[:min(len(row), n_keywords)]]
        doc = ' '.join(row)
        result.append(doc)
    return pd.Series(result, dtype=str)
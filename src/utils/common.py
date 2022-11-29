from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np

# Generate a binary mask to filter out target classes with a total count 
# less than n.
def mask_low_frequency(column, n):
    vals, counts = np.unique(column, return_counts=True)
    return np.isin(column, vals[counts >= n])

def filter_null(data, column):
    return data[data[column].notnull()]

def filter_neq(data, column, value):
    return data[data[column].ne(value)]

def tokenize_all(X):
    all_tokens = []
    n_cols = X.shape[1]
    for i in range(n_cols):
        col = X[:,i]
        for item in col:
            if not item:
                continue
            tokens = word_tokenize(item)
            all_tokens.extend(tokens)
    return all_tokens

def count_stopwords(X):
    tokens = tokenize_all(X)
    stop = set(stopwords.words('english'))
    stop_count = 0
    for token in tokens:
        if token in stop:
            print(token)
            stop_count += 1
    return stop_count

def extract_words_by_tag(text, tag, spacy_nlp):
    if text is None:
        return ''
    doc = spacy_nlp(str(text))
    words = []
    for word in doc:
        if word.pos_.lower() == tag.lower():
            words.append(word.text)
    return ' '.join(words)

def remove_duplicates(text):
    if text is None:
        return ''
    return ' '.join(list(set(text.split())))

def join_string(a, b):
    if a is None and b is not None:
        return b
    elif b is None and a is not None:
        return a
    elif a is None and b is None:
        return ''
    else:
        return ' '.join((a, b))
    
# Builds a dictionary object that maps hash values to keywords using a
# dataframe constructed from the keyword tables provided by lightship.
def load_hash_table(hash_df):
    hash_table = {}
    for row in hash_df.itertuples(index=False):
        hash_table[row.token_hash] = row.token
    return hash_table

# Returns a hash table constructed from multiple hash dataframes.
def load_hash_tables(hash_dfs):
    final_table = {}
    for df in hash_dfs:
        hash_table = load_hash_table(df)
        final_table.update(hash_table)
    return final_table

# Use a hash table created by load_hash_table to unhash a hashed string.
def unhash_str(hashed, hash_table):
    if not isinstance(hashed, str):
        return None
    tokens = hashed.split(' ')
    unhashed = []
    for token in tokens:
        if token in hash_table:
            unhashed.append(hash_table[token])
    return ' '.join(unhashed)

# Unhashes an entire column of a dataframe. Returns the unhashed column as
# a list.
def unhash_column(hash_table, dataframe, column):
    unhashed = []
    for row in dataframe.itertuples(index=False):
        unhashed.append(unhash_str(str(row._asdict()[column]), hash_table))
    return unhashed

# Returns a dictionary that maps account ids to names
def get_account_name_dict(acc_df):
    acc_dict = {}
    for row in acc_df.itertuples():
        acc_dict[row.id] = row.name
    return acc_dict


# Build a sorted list of recommendations from probabilities.
def get_acc_recommendations(probs, ids, acc_name_dict):
    recs = [{'probability': probs[i],
             'account_id': ids[i],
             'account_name': acc_name_dict[ids[i]]}
            for i in range(len(probs))]
    recs.sort(key=lambda k: k['probability'], reverse=True)
    return recs


# Returns a dictionary that maps task ids to a list of team ids
def get_task_team_dict(task_team_df):
    task_team_dict = {}
    for row in task_team_df.itertuples():
        if row.task_id in task_team_dict:
            task_team_dict[row.task_id].append(row.team_id)
        else:
            task_team_dict[row.task_id] = [row.team_id]
    return task_team_dict
      
      
# Returns a dictionary that maps team ids to team names
def get_team_name_dict(team_df):
    name_dict = {}
    for row in team_df.itertuples():
        name_dict[row.id] = row.name
    return name_dict
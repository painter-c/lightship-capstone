## unhash_data.py
##
## Author: Cody Painter
## Date: 2022/10/09
##
## Description:
## Conversion of hashed lightship data to unhashed data frames.

import pandas as pd
from csv_loader import load_lightship_data


# Builds a dictionary object that maps hash values to keywords using a
# dataframe constructed from the keyword tables provided by lightship.
def load_hash_table(hash_df):
    hash_table = {}
    for row in hash_df.itertuples(index=False):
        hash_table[row.token_hash] = row.token
    return hash_table


# Use a hash table created by load_hash_table to unhash a hashed string.
def unhash_str(hash_table, hashed):
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
        unhashed.append(unhash_str(hash_table, str(row._asdict()[column])))
    return unhashed


# Unhashes the title and details columns of the dataframe corresponding to
# task.csv.  Returns a new dataframe with three columns: [title, details,
# task_id]
def unhash_task_df(title_kw_df, details_kw_df, task_df):
    title_kw_dict = load_hash_table(title_kw_df)
    details_kw_dict = load_hash_table(details_kw_df)

    unhashed_obj = {}
    unhashed_obj['title'] = unhash_column(title_kw_dict, task_df, 'title')
    unhashed_obj['details'] = unhash_column(details_kw_dict, task_df, 'details')

    task_df_unhashed = pd.DataFrame(unhashed_obj)
    task_df_unhashed['task_id'] = task_df['id']

    return task_df_unhashed


# Unhashes the content column of the dataframe corresponding to task_event.csv.
# Return a new dataframe with the columns: [content, task_id]
def unhash_task_comment_event_df(task_comment_event_kw_df, task_event_df):
    task_comment_event_kw_dict = load_hash_table(task_comment_event_kw_df)

    unhashed_obj = {}
    unhashed_obj['content'] = unhash_column(task_comment_event_kw_dict, task_event_df, 'content')

    # TODO attach task id to comment events
    return _


################
# Example usages
################

ls_data = load_lightship_data(['task.csv',
                               'task_event.csv',
                               'task_comment_event_keyword_hashes.csv',
                               'task_details_keyword_hashes.csv',
                               'task_title_keyword_hashes.csv'])

task_df_unhashed = unhash_task_df(ls_data['task_title_keyword_hashes'],
                                  ls_data['task_details_keyword_hashes'],
                                  ls_data['task'])

print(task_df_unhashed.head())






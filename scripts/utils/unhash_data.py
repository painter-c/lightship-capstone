## unhash_data.py
##
## Author: Cody Painter
## Date: 2022/10/09
##
## Description:
## Conversion of hashed lightship data to unhashed data frames.

import pandas as pd

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


### Unhashes the content column of the dataframe corresponding to task_event.csv.
### Return a new dataframe with the columns: [content, task_id]
##def unhash_task_comment_event_df(task_comment_event_kw_df, task_event_df):
##    task_comment_event_kw_dict = load_hash_table(task_comment_event_kw_df)
##    task_comment_event_df = filter_task_events(task_event_df, 'CommentEvent',
##                                               ['task_id', 'content'])
##
##    unhashed_obj = {}
##    unhashed_obj['task_id'] = task_comment_event_df['task_id'].tolist()
##    unhashed_obj['content'] = unhash_column(task_comment_event_kw_dict,
##                                            task_comment_event_df,
##                                            'content')
##    return pd.DataFrame(unhashed_obj)
    


################
# Example usages
################

##ls_data = load_lightship_data(['task.csv',
##                               'task_event.csv',
##                               'task_comment_event_keyword_hashes.csv',
##                               'task_details_keyword_hashes.csv',
##                               'task_title_keyword_hashes.csv'])
##
##task_df_unhashed = unhash_task_df(ls_data['task_title_keyword_hashes'],
##                                  ls_data['task_details_keyword_hashes'],
##                                  ls_data['task'])
##
##print(task_df_unhashed.head())
##
##comments_unhashed = unhash_task_comment_event_df(ls_data['task_comment_event_keyword_hashes'],
##                                                 ls_data['task_event'])
##
##print(comments_unhashed.head())



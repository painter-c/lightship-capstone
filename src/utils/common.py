from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import src.utils.loading as loader

import numpy as np

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

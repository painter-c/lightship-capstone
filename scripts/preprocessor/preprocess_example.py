import pandas as pd
import numpy as np

# Task Columns:
#   ['id', 'created', 'modified', 'confidential', 'due_date', 'priority',
#    'project_id', 'state', 'state_modified', 'assignee_id', 'creator_id',
#    'numerator', 'title', 'details']

task_dtypes = {'id':str, 'confidential':str, 'priority':int, \
               'project_id':str, 'state':str, 'assignee_id':str, 'creator_id':str, \
               'numerator':int, 'title':str, 'details':str}

# Load task.csv from lightship-capstone\data\set_1
df = pd.read_csv("../../data/set_1/task.csv", dtype=task_dtypes, parse_dates=[1,2,4,8])

# Drop useless columns
df.drop(['confidential', 'numerator', 'title', 'details'])

print(df.keys())


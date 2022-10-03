import pandas as pd
import numpy as np

DATA_PATH = '../../data/'

# Task Columns:
#   ['id', 'created', 'modified', 'confidential', 'due_date', 'priority',
#    'project_id', 'state', 'state_modified', 'assignee_id', 'creator_id',
#    'numerator', 'title', 'details']

task_dtypes = {'id':str, 'confidential':str, 'priority':int, \
               'project_id':str, 'state':str, 'assignee_id':str, 'creator_id':str, \
               'numerator':int, 'title':str, 'details':str}

# Load task.csv from lightship-capstone\data\set_1
df = pd.read_csv(DATA_PATH + 'set_1/task.csv', dtype=task_dtypes, parse_dates=[1,2,4,8])

# Drop useless columns
# df.drop(['confidential', 'numerator'])

# See if there are duplicate titles
c1, u1 = pd.factorize(df.loc[:,'title'])
c2, u2 = pd.factorize(df.loc[:,'details'])

new_df = df.assign(title=c1, detail=c2)
new_df.to_csv('test')

print('titles:', len(c1), len(u1))
print('details:', len(c2), len(u2))

print('# of null details:', df.loc[:,'details'].isna().sum())

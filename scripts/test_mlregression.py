import pandas as pd
import numpy as np

from utils.csv_loader import load_lightship_data

## test_mlregression.py
##
## Use the assignee_id and project_id to predict team_observers

lightship_data = load_lightship_data()

print('Loaded files:')
for key in lightship_data.keys():
    print('  ', key+'.csv')

task_df = lightship_data['task'].copy()
task_teams_df = lightship_data['task_teams'].copy()

# Drop columns of non-interest
task_df.drop(['created',
              'modified',
              'confidential',
              'due_date',
              'priority',
              'state',
              'state_modified',
              'creator_id',
              'numerator',
              'title',
              'details',
              ], axis=1, inplace=True)

# Drop rows where assignee_id is missing
task_df = task_df[task_df['assignee_id'].notnull()]

# Rename id column for join
task_df.rename(columns={'id': 'task_id'}, inplace=True)

# Join the the two tables and drop the index
merged_df = pd.merge(task_df, task_teams_df, on='task_id', how='outer')
merged_df.drop('task_id', axis=1, inplace=True)


# Drop null rows
merged_df = merged_df[merged_df['project_id'].notnull()]
merged_df = merged_df[merged_df['assignee_id'].notnull()]
merged_df = merged_df[merged_df['team_id'].notnull()]

# Convert string ids to unique ints
merged_df['project_id'] = pd.factorize(merged_df['project_id'])[0]
merged_df['assignee_id'] = pd.factorize(merged_df['assignee_id'])[0]

# Separate the target column
y = merged_df['team_id']
merged_df.drop('team_id', axis=1, inplace=True)

# Test logistic regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(merged_df, y, train_size=3/10)

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train)

X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression()
model = clf.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

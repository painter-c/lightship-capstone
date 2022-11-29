import src.utils.config as config
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils.loading import load_lightship_data

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

cf = config.load()

ls = load_lightship_data(cf, ['task.csv', 'account.csv'])

df = ls['task']

# convert assignee_ids to names
def convert_name(name):
    return '_'.join(name.lower().split())
    
accounts = ls['account']
accounts['name'] = accounts['name'].map(convert_name)

name_lookup = dict()
for row in accounts.itertuples():
    name_lookup[row.id] = row.name

df['assignee_id'] = df['assignee_id'].map(name_lookup)
df['creator_id'] = df['creator_id'].map(name_lookup)

# remove entries with no assignee
df = df[df['assignee_id'].notnull()]


n_automated = df[df['creator_id'].eq('lightship_automation')].shape[0]
ratio = n_automated / df.shape[0]
print(f'%{ratio * 100} of task assignments were automated.')
# filter out automated task assignments
df = df[df['creator_id'].ne('lightship_automation')]

# plot number of tasks per assignee
assignee_counts = dict()
for row in df.itertuples():
    if row.assignee_id in assignee_counts:
        assignee_counts[row.assignee_id] += 1
    else:
        assignee_counts[row.assignee_id] = 1
        
names  = np.array(list(assignee_counts.keys()))
counts = np.array(list(assignee_counts.values()))

data = np.hstack((names.reshape(-1,1), counts.reshape(-1,1)))

# plt.bar(names, height=counts)
# plt.xticks(rotation=45, ha='right')

df['created'] = df['created'].map(lambda t: t.timestamp())

scaler = MinMaxScaler()
df[['created']] = scaler.fit_transform(df[['created']])

encoder = OrdinalEncoder()
df[['project_id', 'creator_id']] = encoder.fit_transform(df[['project_id', 'creator_id']])
df[['assignee_id']] = encoder.fit_transform(df[['assignee_id']])

plt.clf()
#plt.tick_params(labelleft=False)
plt.xlabel('time')
plt.ylabel('project_id')
plt.scatter(df['created'], df['project_id'], c=df['project_id'])
plt.colorbar()
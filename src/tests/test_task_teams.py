# Classification of task teams

import src.utils.csv_loader as csv
import src.utils.config as cfg
import src.utils.util_misc as misc
import src.utils.unhash_data as ud
from src.transforms import team_target_transform
from src.pipelines import build_pipeline_A, build_pipeline_bagofwords

from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

config = cfg.load()

all_data = csv.load_lightship_data(config,
                                   ['task.csv',
                                    'task_teams.csv',
                                    'task_title_keyword_hashes.csv',
                                    'task_details_keyword_hashes.csv',
                                    'team.csv',
                                    'task_teams.csv'])

task_data = all_data['task']

# Build a dataset where examples with multiple team observers are duplicated, 
# one team observer per example.
task_team_dict = misc.get_task_team_dict(all_data['task_teams'])
task_data = team_target_transform(task_data, task_team_dict)

# Extract the target column
y = task_data['team_id']
task_data.drop('team_id', axis=1, inplace=True)

# Load keyword hash dictionary for text pipeline
keyword_dict = ud.load_hash_tables((all_data['task_title_keyword_hashes'],
                                    all_data['task_details_keyword_hashes']))

tsk_pipe = make_pipeline(build_pipeline_A(), LogisticRegression(max_iter=1000))

bow_pipe = make_pipeline(build_pipeline_bagofwords(keyword_dict),
                         LogisticRegression(max_iter=1000))

clf = StackingClassifier([
    ('tsk', tsk_pipe),
    ('bow', bow_pipe)
])

scores = cross_val_score(clf, task_data, y, scoring='accuracy', cv=5)
print(scores.mean())
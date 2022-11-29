import src.utils.config as config
import src.utils.common as common
import src.utils.loading as loading
import src.pipelines as pipelines

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

import numpy as np

cfg = config.load()

REQUIRED_FILES = ['task.csv']

data = loading.load_lightship_data(cfg, REQUIRED_FILES)

task_data = data['task']

# Initial preprocessing
# 1. Filter null targets
task_data = common.filter_null(task_data, 'assignee_id')

# 2. Filter automated entries
task_data = common.filter_neq(task_data, 'creator_id', cfg['automated_account_id'])

# 3. Filter low frequency target classes (< 5)
mask = common.mask_low_frequency(task_data['assignee_id'], 5)
task_data = task_data[mask]

# Randomized search

# Create the random grid
param_grid = {
    'clf': [RandomForestClassifier()],
    'clf__n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
    'clf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)] + [None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__bootstrap': [True, False]
}

rf_random = RandomizedSearchCV(
    estimator=pipelines.pipeline_task(),
    param_distributions=param_grid,
    n_iter=100,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1)

y = task_data['assignee_id']

rf_random.fit(task_data, y)
print(rf_random.best_params_)
print(rf_random.best_score_)
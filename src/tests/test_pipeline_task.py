import src.utils.config as config_manager
import src.pipelines as pipelines
import src.transformers as tfm
from src.utils.loading import LightshipLoader

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

import numpy as np

config = config_manager.load()
loader = LightshipLoader(config)

task_df = loader.load('task.csv')
creator_blacklist = config_manager.load_creator_blacklist()
assignee_blacklist = config_manager.load_assignee_blacklist()

# Preprocessing
preprocess = make_pipeline(
    tfm.NullEntryFilter('assignee_id'),
    tfm.BlacklistFilter(creator_blacklist, 'creator_id'),
    tfm.BlacklistFilter(assignee_blacklist, 'assignee_id'),
    tfm.LowFrequencyFilter('assignee_id', config['min_class_frequency']),
)

task_df = preprocess.fit_transform(task_df)

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

y = task_df['assignee_id']

rf_random.fit(task_df, y)
print(rf_random.best_params_)
print(rf_random.best_score_)
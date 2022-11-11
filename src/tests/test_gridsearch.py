from sklearn.preprocessing import StandardScaler

import src.utils.config as config

from src.utils.csv_loader import load_lightship_data

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split

cfg = config.load()

df = load_lightship_data(cfg['datasets']['set_1'],
                         ['task.csv'])

df = df['task']
df = df[df['assignee_id'].notnull()]

X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', random_state=1))
param_grid_rfc = [{
    'randomforestclassifier__max_depth': [2, 3, 4],
    'randomforestclassifier__max_features':[2, 3, 4, 5, 6]
}]

gridS_RF = GridSearchCV(estimator=pipeline, param_grid=df, #unsure
    scoring='accuracy', cv=10, refit=True, n_jobs=1)

gridS_RF = gridS_RF.fit(X_train, y_train)

print(gridS_RF.best_score_)
print(gridS_RF.best_params_)

bestRFC = gridS_RF.best_estimator_
print('Test accuracy: %.3f' % bestRFC.score(X_test, y_test))

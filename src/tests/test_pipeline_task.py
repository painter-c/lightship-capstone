import src.utils.config as config

from src.pipelines import build_pipeline_A
from src.utils.csv_loader import load_lightship_data

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import sklearn.linear_model as lm
import sklearn.ensemble as en
import sklearn.svm as sv
import sklearn.neural_network as nn

cfg = config.load()

df = load_lightship_data(cfg,
                         ['task.csv'])

df = df['task']
df = df[df['assignee_id'].notnull()]

y = df['assignee_id']

ests = [
    lm.LogisticRegression(max_iter=1000),
    lm.PassiveAggressiveClassifier(),
    lm.Perceptron(),
    lm.RidgeClassifier(),
    lm.SGDClassifier(),
    en.AdaBoostClassifier(),
    en.BaggingClassifier(),
    en.ExtraTreesClassifier(),
    en.GradientBoostingClassifier(),
    en.RandomForestClassifier(),
    #sv.LinearSVC(max_iter=10000),
    sv.SVC(),
    nn.MLPClassifier(max_iter=1000)
]

score_method = 'accuracy'
est_scores = []
for est in ests:
    pipe = make_pipeline(build_pipeline_A(), est)
    scores = cross_val_score(pipe, df, y, cv=5, scoring=score_method)
    est_scores.append((est.__class__.__name__, scores.mean()))
    #print(f'{est.__class__.__name__} accuracy: {scores.mean()}')

est_scores.sort(key = lambda x: x[1], reverse=True)

print('Accuracy scores:')
for i, entry in enumerate(est_scores):
    print(f'{i}. {entry[0]}: {entry[1]}')

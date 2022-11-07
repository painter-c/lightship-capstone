import src.utils.config as config

from src.pipelines import build_pipeline_A
from src.utils.csv_loader import load_lightship_data

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

cfg = config.load()

df = load_lightship_data(cfg['datasets']['set_1'],
                         ['task.csv'])

df = df['task']
df = df[df['assignee_id'].notnull()]

pipeline = make_pipeline(
    build_pipeline_A(),
    GradientBoostingClassifier()
)

y = df['assignee_id']
scores = cross_val_score(pipeline, df, y, cv=5, scoring='roc_auc_ovo')

print(scores.mean())

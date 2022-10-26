import pipelines
import utils.csv_loader as loader

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

df = loader.load_lightship_data(['task.csv'])['task']

df = df[df['assignee_id'].notnull()]

pipeline = make_pipeline(
    pipelines.build_pipeline_A(),
    GradientBoostingClassifier()
)

y = df['assignee_id']

scores = cross_val_score(pipeline, df, y, cv=5, scoring='roc_auc_ovo')

print(scores.mean())
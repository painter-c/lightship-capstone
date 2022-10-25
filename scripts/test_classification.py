import pandas as pd
import numpy  as np

from utils.csv_loader import load_lightship_data

from sklearn.pipeline import make_union, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import MissingIndicator, SimpleImputer

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import top_k_accuracy_score

# Load task data
df = load_lightship_data(['task.csv'])['task']

# filter columns where the target variable is null
df = df[df['assignee_id'].notnull()]


# Transforms each date in the created column to an integer value representing
# the day of the week (0 - 6).
def dayofweek_transform(df):
    day_col = df['created'].dt.dayofweek
    return pd.DataFrame(day_col)


# Transforms each date in the created column to an integer value representing
# the hour of the day (0 - 23).
def timeofday_transform(df):
    time_col = df['created'].dt.hour
    return pd.DataFrame(time_col)


# Prepares nominal columns for one hot encoding.
#
# Inputs:
#   created: datetime64
#   due_date: datetime64
#   project_id: string
#   creator_id: string
#
# Outputs:
#   day_of_week: integer (nominal)
#   has_due_date: boolean
#   project_id: string (unchanged)
#   project_id: string (unchanged)
#
nom_ct = make_column_transformer(
    (FunctionTransformer(dayofweek_transform), ['created']),
    (MissingIndicator(), ['due_date']),
    ('passthrough', ['project_id', 'creator_id']))


# Prepares ordinal columns.
#
# Inputs:
#   created: datetime64
#   priority: integer
#
# Outputs:
#   time_of_day: integer
#   priority: integer (unchanged)
ord_ct = make_column_transformer(
    (FunctionTransformer(timeofday_transform), ['created']),
    ('passthrough', ['priority']))


# One hot encodes the nominal columns outputed from nom_ct.
#
ohe_pl = make_pipeline(nom_ct, OneHotEncoder(handle_unknown='ignore'))


# Combines the ordinal columns from ord_ct with the one hot encoded 
# columns from ohe_pl.
data_transformer = make_union(ohe_pl, ord_ct)


# MAIN CLASSIFICATION PIPELINE
#
pipeline = make_pipeline(data_transformer,
                         SimpleImputer(strategy='mean'),
                         GradientBoostingClassifier())


#
# RUN CROSS VALIDATION
#

y = df['assignee_id']
scores = cross_val_score(pipeline, df, y, cv=5, scoring='balanced_accuracy')

print(scores.mean())
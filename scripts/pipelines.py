import transforms as T

from sklearn.pipeline import make_union, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import MissingIndicator, SimpleImputer

def _build_pipeline_ohe():
    return make_pipeline(
        make_column_transformer(
            (FunctionTransformer(T.dayofweek_transform), ['created']),
            (MissingIndicator(), ['due_date']),
            ('passthrough', ['project_id', 'creator_id'])
        ),
        OneHotEncoder(handle_unknown='ignore')
    )

def build_pipeline_A():
    return make_pipeline(
        make_union(
            _build_pipeline_ohe(),
            make_column_transformer(
                (FunctionTransformer(T.timeofday_transform), ['created']),
                ('passthrough', ['priority'])
            )
        ),
        SimpleImputer(strategy='mean')
    )
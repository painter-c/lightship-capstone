import src.utils.config as config_manager
import src.transformers as tfm
from src.utils.loading import LightshipLoader

from sklearn.pipeline import make_pipeline

import pandas as pd

config = config_manager.load()
loader = LightshipLoader(config)

test_data = {
    'column_a': ['This is some example data.', 'To test text preprocessing.'],
    'column_b': ['It should, remove puncuation.',
                 'Stop-words as well. Should work on multiple sentences']
}

test_df = pd.DataFrame(test_data, dtype=str)

columns = ['column_a', 'column_b']
pipe = make_pipeline(
    tfm.WordTokenizer(columns),
    tfm.StopwordFilter(columns),
    tfm.WordStemmer(columns),
    tfm.WordTokenJoin(columns)
)

result = pipe.fit_transform(test_df)

print(result.head())
import src.utils.config as config
import src.utils.util_misc as util_misc

from src.pipelines import build_pipeline_A
from src.utils.csv_loader import load_lightship_data

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

cfg = config.load()

ls = load_lightship_data(cfg, ['task.csv', 'account.csv'])

df = ls['task']
df = df[df['assignee_id'].notnull()]
df = df[df['creator_id'].ne('4b5f8672-2180-4507-a694-4926e0da7f83')]

pipe = make_pipeline(
    build_pipeline_A(),
    GradientBoostingClassifier()
)

y = df['assignee_id']
scores = cross_val_score(pipe, df, y, cv=3, scoring='accuracy')
print(scores.mean())

# test mapping probabilities to names
# acc_dict = util_misc.get_account_name_dict(ls['account'])

# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.95)

# pipe.fit(X_train, y_train)

# ps = pipe.predict_proba(X_test)

# for i in range(ps.shape[0]):
#     p_row = ps[i]
#     print(f'Expected: {acc_dict[y_test.iat[i]]}')
#     reccs = util_misc.get_acc_reccomendations(p_row,
#                                               pipe.classes_,
#                                               acc_dict)
#     for j, recc in enumerate(reccs):
#         print(f'{j}. {recc.name} {recc.account_id} {recc.probability:.10f}')
#     print()
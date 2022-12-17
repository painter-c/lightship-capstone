import src.pipelines as pipelines

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

import numpy as np


lreg_param_grid = {
    'clf': [LogisticRegression(class_weight='balanced', max_iter=2000)],
    'clf__C': np.logspace(-4, 4, 8)
}

rfor_param_grid = {
    'clf': [RandomForestClassifier(class_weight='balanced')],
    'clf__n_estimators': [100, 200, 300],
    'clf__max_features': ['sqrt', 'log2'],
    'clf__max_depth': [6, 8, 10]
}

adab_param_grid = {
    'clf': [AdaBoostClassifier()],
    'clf__n_estimators': [10, 50, 100, 500],
    'clf__learning_rate': [0.001, 0.01, 0.1, 0.3]
}


def optimize_pipeline(pipeline, X, y, cv, scoring):
    
    cross_validator = StratifiedKFold(n_splits=cv)
    
    # 1. Logistic regression
    lreg_gridsearch = GridSearchCV(
        pipeline,
        param_grid = lreg_param_grid,
        cv = cross_validator,
        scoring = scoring,
        refit = True
    )
    
    lreg_gridsearch.fit(X, y)
    
    # 2. Random forest
    rfor_gridsearch = GridSearchCV(
        pipeline,
        param_grid = rfor_param_grid,
        cv = cross_validator,
        scoring = scoring,
        refit = True
    )
    
    rfor_gridsearch.fit(X, y)
    
    # 3. Adaboost
    adab_gridsearch = GridSearchCV(
        pipeline,
        param_grid = adab_param_grid,
        cv = cross_validator,
        scoring = scoring,
        refit = True
    )
    
    adab_gridsearch.fit(X, y)
    
    results = [
        (lreg_gridsearch.best_estimator_, lreg_gridsearch.best_params_, lreg_gridsearch.best_score_),
        (rfor_gridsearch.best_estimator_, rfor_gridsearch.best_params_, rfor_gridsearch.best_score_),
        (adab_gridsearch.best_estimator_, adab_gridsearch.best_params_, adab_gridsearch.best_score_),
    ]

    return max(results, key = lambda x: x[2])[0:2]
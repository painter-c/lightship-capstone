# Local imports
import src.utils.config as config_manager
import src.utils.loading as loading
import src.transformers as transformers
import src.pipelines as pipelines
import src.utils.reporting as reporting
from src.optimizer import optimize_pipeline
# Python imports
import sys
import os.path as path
import pickle
# Gensim imports
import gensim.downloader as gensim_api
import gensim.models as gensim_models
# Scikit learn imports
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, cross_validate


REQUIRED_FILES = ['task_title_keyword_hashes.csv',
                  'task_details_keyword_hashes.csv',
                  'account.csv',
                  'task.csv']

CACHED_MODEL_NAME = 'cached-model'

def exit_fatal(error):
    print('Error:', error)
    sys.exit()


def check_optimize_flag():
    return '-optimize' in sys.argv


def check_usedefault_flag():
    return '-usedefault' in sys.argv


def load_optimized_model(config):
    file_path = config['cache_location'] + CACHED_MODEL_NAME + '.pickle'
    return pickle.load(open(file_path, 'rb'))
   
 
def save_optimized_model(model, config):
    file_path = config['cache_location'] + CACHED_MODEL_NAME + '.pickle'
    pickle.dump(model, open(file_path, 'wb'))


def check_optimized_model(config):
    file_path = config['cache_location'] + CACHED_MODEL_NAME + '.pickle'
    return path.exists(file_path)


def check_required_files(data_path, keyword_path):
    has_missing = False
    for file in REQUIRED_FILES:
        in_data_path = path.exists(data_path + file)
        in_keyword_path = path.exists(keyword_path + file)
        if not in_data_path and not in_keyword_path:
            print(f'Missing file: {data_path+file}')
            has_missing = True
    if has_missing:
        exit_fatal(f'Error: there were missing files at "{data_path}"')


def load_required_files(config):
    
    data_path = config['data_path']
    if not path.exists(data_path):
        exit_fatal(f'Data path "{data_path}" was not found.')
    
    keyword_path = config['keyword_path']
    if not path.exists(keyword_path):
        exit_fatal(f'Keyword path "{keyword_path}" was not found.')
        
    check_required_files(data_path, keyword_path)
    return loading.load_lightship_data(config, REQUIRED_FILES)


# Attempts to load the gensim word vectorization model from the local model 
# cache, otherwise downloads it from the gensim api.
def load_gensim_model(config):
    model_path = config['cache_location'] + config['kv_model'] + '.kv'
    if path.exists(model_path):
        return gensim_models.KeyedVectors.load(model_path)
    # No cached model found, attempt to download using the gensim api
    try:
        model = gensim_api.load(config['kv_model'])
        model.save(model_path)
        return model
    except Exception:
        exit_fatal(f'"{config["kv_model"]}" is not a valid Gensim model.')


# This runs the data through a preprocessing pipline that will prepare raw 
# text data for machine learning. Should only be used when unhashing is 
# disabled, signaling that the input data is in full sentence form.
def text_preprocessing(input_data):
    columns = ['title', 'details']
    preprocess = make_pipeline(
        transformers.WordTokenizer(columns),
        transformers.StopwordFilter(columns),
        transformers.WordTokenJoin(columns)
    )
    return preprocess.fit_transform(input_data)


def stem_words(input_data):
    columns = ['title', 'details']
    preprocess = make_pipeline(
        transformers.WordTokenizer(columns),
        transformers.WordStemmer(columns),
        transformers.WordTokenJoin(columns)
    )
    return preprocess.fit_transform(input_data)


# Preprocessing steps that always happen before entering the evaluation 
# pipeline.
def initial_preprocessing(input_data, keyword_table, config, creator_blacklist, assignee_blacklist):
    steps = []
    steps.append(transformers.NullEntryFilter('assignee_id'))
    steps.append(transformers.BlacklistFilter(creator_blacklist, 'creator_id'))
    steps.append(transformers.BlacklistFilter(assignee_blacklist, 'assignee_id'))
    steps.append(transformers.LowFrequencyFilter('assignee_id', config['min_class_frequency']))
    
    if config['unhashing_enabled']:
        steps.append(transformers.HashDecoder(keyword_table, ['title', 'details']))
        
    preprocess = make_pipeline(*steps)
    input_data = preprocess.fit_transform(input_data)
    
    if not config['unhashing_enabled']:
        print('Hash decoding was disabled. Performing further text preprocessing.')
        print(' (tokenization, stopword removal, punctuation removal, etc...)')
        input_data = text_preprocessing(input_data)
    
    if config['stemming_enabled']:
        input_data = stem_words(input_data)
    
    return shuffle(input_data)


# A default pipeline with no optimization that will be constructed when the
# grid search option is disabled in the config.
def configure_default_pipeline(kv_model, config):
    
    pipes = []

    if config['task_pipe_enabled']:
        pipes.append(('task', pipelines.pipeline_task()))

    if config['wordvec_pipe_enabled']:
        pipes.append(('word', pipelines.pipeline_word_vectorizer(kv_model)))

    if config['countvec_pipe_enabled']:
        pipes.append(
            ('count', pipelines.pipeline_count_vectorizer()))
        
    return StackingClassifier(pipes)


# Runs a grid search on each sub pipeline (task, count vec., word vec.) and 
# contructs a stacking classifier with the best parameters for each.
def configure_optimized_pipeline(X, y, cv, scoring, kv_model, config):
    
    best_task_clf, best_task_params = optimize_pipeline(
        pipelines.pipeline_task(), X, y, cv, scoring)
    
    best_word_clf, best_word_params = optimize_pipeline(
        pipelines.pipeline_word_vectorizer(kv_model), X, y, cv, scoring)
    
    best_count_clf, best_count_params = optimize_pipeline(
        pipelines.pipeline_count_vectorizer(), X, y, cv, scoring)

    pipes = []
    
    if config['task_pipe_enabled']:
        pipes.append(
            ('task', best_task_clf))

    if config['wordvec_pipe_enabled']:
        pipes.append(
            ('word', best_word_clf))

    if config['countvec_pipe_enabled']:
        pipes.append(
            ('count', best_count_clf))
    
    best_clfs = {
        'task': best_task_clf,
        'word': best_word_clf,
        'count': best_count_clf
    }
    
    best_params = {
        'task': best_task_params,
        'word': best_word_params,
        'count': best_count_params
    }
    
    return (StackingClassifier(pipes), best_clfs, best_params)


def main():

    config = config_manager.load()
    loader = loading.LightshipLoader(config)
    account_lookup = loader.load_account_lookup()
    kv_model = load_gensim_model(config)
    
    frames = load_required_files(config)
    
    input_data = frames['task']
    print('Preprocessing data...')
    input_data = initial_preprocessing(
        input_data,
        loader.load_keyword_table(),
        config,
        config_manager.load_creator_blacklist(),
        config_manager.load_assignee_blacklist())
    
    print('Running model evaluation...')
    if check_optimize_flag():
        grid_search_evaluation(input_data, kv_model, config, account_lookup)
    else:
        default_evaluation(input_data, kv_model, config, account_lookup)


def default_evaluation(input_data, kv_model, config, account_lookup):
    
    X = input_data
    y = input_data['assignee_id']
    
    clf = None
    if check_optimized_model(config) and not check_usedefault_flag():
        clf = load_optimized_model(config)
        print('A cached optimized model was loaded.')
    else:
        clf = configure_default_pipeline(kv_model, config)
        print('Using the default pipeline.')
    
    report_builder = reporting.ReportBuilder(min_spacing=10)
    
    run_cross_validation(
        X, y, clf, kv_model, config, report_builder)
    
    run_recommendations(
        X, y, clf, kv_model, config, account_lookup, report_builder)
    
    report_builder.data_summary(X, config)
    report_builder.configuration(config)
    report_builder.print_report()
    report_builder.save_report(config['report_location'])


def grid_search_evaluation(input_data, kv_model, config, account_lookup):
    
    X = input_data
    y = input_data['assignee_id']
    
    report_builder = reporting.ReportBuilder(min_spacing=10)
    
    clf = run_optimization(X, y, kv_model, config, report_builder)
    
    run_cross_validation(
        X, y, clf, kv_model, config, report_builder)
    
    run_recommendations(
        X, y, clf, kv_model, config, account_lookup, report_builder)
    
    report_builder.data_summary(input_data, config)
    report_builder.configuration(config)
    report_builder.print_report()
    report_builder.save_report(config['report_location'])
    

def run_optimization(X, y, kv_model, config, report_builder):
    
    print('Optimizing model parameters using grid search. This may take a while...')
    optimized_pipeline, best_clfs, best_params = configure_optimized_pipeline(
        X, y, config['cv_folds'], config['cv_scoring'], kv_model, config)
    
    report_builder.grid_search(best_clfs, best_params, config)
    
    save_optimized_model(optimized_pipeline, config)
    print('Optimized model has been cached.')
    
    return optimized_pipeline


def run_cross_validation(X, y, clf, kv_model, config, report_builder):
    
    cv = StratifiedKFold(n_splits=config['cv_folds'])
    
    scoring = ['accuracy', config['cv_scoring']]
    
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

    report_builder.cross_validation(
        scores, {'Accuracy':'accuracy', 'Roc auc ovr':'roc_auc_ovr'})
    

def run_recommendations(X, y, clf, kv_model, config, account_lookup, report_builder):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = config['test_size'])

    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    y_class = clf.classes_
    task_ids = X['id'].to_numpy()

    report_builder.classification(y_pred, y_test, y_prob)

    report_builder.recommendations(
        task_ids, y_test, y_prob, y_class, account_lookup, config)


if __name__ == '__main__':
    main()
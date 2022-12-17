# Lightship Capstone Project
## Driver Program

### Prerequisites

To run this program you must have Python 3 installed on your system. The following python packages will be necessary.

```
pip install numpy
pip install pandas
pip install sklearn
pip install gensim
pip install nltk
pip install tomli
pip install scipy
pip install bertopic
pip install umap
```

The script *install_deps.py* in the project folder will automatically install the above packages when run.

### Run configuration

There are a number of configuration options in config.toml that can be used to change the behaviour of the classification pipeline.

#### data_path

The path to the dataset that will be used by the driver program. The path is relative to the top level project directory.

#### keyword_path

The path to the keyword data used to decode the hashed title and description columns.

#### kv_model

The name of the Gensim KeyedVector model. This model is loaded when the word vectorization pipeline is enabled. The value can be one of:

```
fasttext-wiki-news-subwords-300
conceptnet-numberbatch-17-06-300
word2vec-ruscorpora-300
word2vec-google-news-300
glove-wiki-gigaword-50
glove-wiki-gigaword-100
glove-wiki-gigaword-200
glove-wiki-gigaword-300
glove-twitter-25
glove-twitter-50
glove-twitter-100
glove-twitter-200
```

#### unhashing_enabled

Controls whether the title and details columns will be unhashed before classification is attempted. If this is false the program will assume it is dealing with unhashed data and therefore perform extra text preprocessing steps such as tokenization, stopword removal, and stripping punctuation.

#### task_pipe_enabled

Enables or disables the basic task classification pipeline. This pipeline uses the *creator_id* and *project_id* columns as a one-hot-encoded binary matrix.

#### wordvec_pipe_enabled

Enables or disables the word vectorization pipeline. This pipline uses the *title* and *description* columns encoded as word vectors. The word vectors in this pipeline are generated from the KeyedVector model specified by the *kv_model* configuration option. Note that the word vectors generated by the KeyedVector model can be quite large and add significantly to the training time of the classification pipeline.

#### countvec_pipe_enabled

Enables or disables the count vectorization pipeline. This pipeline uses the *title* and *description* columns encoded as a weighted term frequency matrix generated by the TF/IDF technique.

#### stemming_enabled

If this option is enabled, words in the *title* and *description* columns are stemmed. Meaning word suffixes are removed to product a word in base form. This allows the same word with different inflections to be treated equivalenty by the count vectorization pipeline. Note that the algorithm used can produce base words that are not real english words such as *leav* instead of *leave*. It should therefore only be used when word vectorization is disabled, otherwise many of the stemmed words will not be found in the KeyedVector model's dictionary.

#### test_size

This controls the size of the test set when the program generates assignment recommendations only. The optimization and cross validation stages of the program use K fold cross validation and therefore are not effected by this value.

#### cv_folds

The number of folds used in K fold cross validation.

#### cv_scoring

The scoring metric used by K fold cross validation. The possible scoring function can be found in Sklearn's [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html). Note that the scoring metric here must support multiclass classification.

#### min_class_frequency

The minimum number of examples a certain target class *(ie. assignee)* must have to be included in the input data. The value of this should be at least greater than the *cv_folds* parameter so there can be at least one example of each target class in each fold of K fold cross validation. If the value of *cv_folds* is too high and there are low frequency classes in the input data, Sklearn may fail during model fitting.

#### max_recommendations

The maximum number of recommendations that are shown per task in the output report.

### Assignee and Creator Blacklisting

To instruct the program to ignore certain account ids as creators or assignees, simple paste the target account id into either the *creator_blacklist.txt* or *assignee_blacklist.txt* file. Those corresponding entries will then be filtered out of the data during preprocessing. There is one id already in the creator blacklist that corresponds to the lightship automation account to filter out the automated tasks. If using a different dataset the id of the lightship automation account for that data should be put into the creator blacklist as well.

### Running the program

The program is structured as a python module. To run it you must open a terminal in the root directory of the project and run the following command.

```
python -m run_pipeline
```

Running the program with no command line options will result in two different outcomes. If there is a cached optimized classification pipeline from a previous run it will load and use it. This is useful when you want to run a pipeline again without recomputing the optimal model, which can take a much longer time. To run the program with optimization you can run the following command.

```
python -m run_pipeline -optimize
```

This will automatically find a good model for the particular input configuration and cache it for further use. Note that if you wish to run the default pipeline while a there and optimized model that has been cached, you must either delete the cached model from the cache/ directory or run the program with the following command line argument.

```
python -m run_pipeline -usedefault
```

Upon running the program will generate a classification report showing the results of optimization, cross validation, and train-test classification. The report is printed to the console and saved in a .txt file in the reports/ directory. The report file also includes a visualization of the recommendation for a given run.

### Using the Program with Different Data

To run the program with different input data, the new data should first be prepared in the same manner as the existing data. The data should be then put in its own folder in the data/ directory. The *data_path* configuration option in config.toml should be set to point to the new dataset. If the *title* and *description* columns in the new data are not hashed, then the *unhashing_enabled* option should be set to *false* so that the text is preprocessed correctly. Once these steps are done the program should now run using the new data.

### Interpreting the Classification Report

It is important to distinguish between the cross validation and classification results in the generated report. Cross validation is a technique for evaluating the performance of a machine learning model. It divides the dataset into a number of bins called folds. It then iteratively chooses one fold as a test set and the rest as a train set, then evaluates the model with that particular split. The scores for each split are then averaged to produce the final score. This produces a more stable score that doesn't change as much from run to run. The classification results section is the result of evaluation on only one train/test split that is determined by the *test_size* configuration option in config.toml. This result is much less stable and may vary by quite a bit from run to run. It is this train/test split that is used to generate the recommendation visualizations in the recommendations section.

If the program is run in optimization mode, the report will also include and optimization section that shows the hyperparameters of the best classifier for each sub-pipeline in the classifier stack. The models shown will be one of LogisticRegression, RandomForestClassifier, or AdaBoostedClassifier; each with a number of possible hyperparameters. This optimization is just an approximation of the best model and may vary from run to run. However it will generate an acceptable model each time.

The main scoring metrics used for cross validation are roc_auc_ovr and accuracy. Roc auc is used because it is less sensitive to class imbalance. The ovr stands for one-vs-rest and is a method of adapting roc auc (Receiver Operator Characteristic Area Under the Curve) to multiclass problems. The accuracy score can be misleading in highly unbalanced datasets because a classifier could just guess the most frequent target class to receive a good accuracy score. Roc auc ranges from 0.5 to 1.0, where 0.5 is the worst possible model and 1.0 is the best. Roc auc is also the metric used during the optimization routine to find the best model.

### Running Test Files

To run one of the individual test files in the tests/ directory the following command should be used. (Without the .py extension)

```
python -m src.tests.name_of_test
```

There are two main tests of interest in the tests/ directory. The first one is test_bertopic, which is an example of topic analysis using Latent Dirichlet Analysis. Running this file will show a list of arbritrary topics that have been generated from the current dataset specified by the *data_path* config option. The second test of interest is test_task_teams, which is an example of classifying teams as observers rather than individuals as assignees. While this classification task was not included in the driver program, the process is nearly identical aside from two differences. One being that with team classification, the assignee_id column is used a a predictor. The other difference is that a single task can have multiple teams associated with it, therefore the input data must be flattened so that each instance only has one associated team. The same principle is true for individuals as observers, however we were unable to experiment with this due to lack of data.
import fire
from nlp_test_neoway import config, nlp_utils  # noqa  change for packaging
import os
from os import path

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('rslp')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import joblib
from unidecode import unidecode


def features(data_path, **kwargs):
    """Function that will generate the dataset for your model. It can
    be the target population, training or validation dataset. You can
    do in this step as well do the task of Feature Engineering.

    data_path: str (path to a .csv file)

    NOTE
    ----
    config.data_path: workspace/data

    You should use workspace/data to put data to working on.  Let's say
    you have workspace/data/iris.csv, which you downloaded from:
    https://archive.ics.uci.edu/ml/datasets/iris. You will generate
    the following:

    + workspace/data/test.csv
    + workspace/data/train.csv
    + workspace/data/validation.csv
    + other files

    With these files you can train your model!
    """
    print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")

    df = pd.read_csv(data_path)

    df = nlp_utils.data_cleaning(df.copy())

    train,test = train_test_split(df, test_size=0.2, random_state=42) # type: ignore

    train.to_csv(path.join(config.data_path, 'train.csv'))
    test.to_csv(path.join(config.data_path, 'test.csv'))

    print("==> Datasets generated and saved on:", '\n', path.join(config.data_path, 'train.csv'), '\n', path.join(config.data_path, 'test.csv'))


def train(**kwargs):
    """Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.
    """
    print("==> TRAINING YOUR MODEL!")

    pt_stopwords = stopwords.words('portuguese')

    opt_svc = Pipeline([('tfidf', TfidfVectorizer(lowercase=True, min_df=5, max_df=0.85, stop_words=pt_stopwords, max_features=5000, ngram_range=(1, 3), use_idf=False)),
                     ('svc', LinearSVC(class_weight='balanced'))])

    parameters_svc = {'svc__C':[0.1,1,10,100,1000],
                    'svc__loss':['hinge', 'squared_hinge'], 
    }

    precision = metrics.make_scorer(metrics.precision_score, pos_label="Yes")
    svc_grid = GridSearchCV(opt_svc, parameters_svc, cv=2, verbose=2, scoring=precision, n_jobs=-1, refit=True)

    df_train = pd.read_csv(path.join(config.data_path, 'train.csv'))
    df_train.dropna(inplace=True)
    df_train = df_train.astype(str)
    X_train = df_train['review_text']
    y_train = df_train['recommend_to_a_friend']
    svc_grid.fit(X_train, y_train)

    best_precision = svc_grid.best_score_
    best_parameters = svc_grid.best_params_

    print("Best precision: {}".format(best_precision))
    print("Best Parameters:", best_parameters)

    # Scoring Training data
    print('\n', 'Precision on training data: ', round(svc_grid.score(X_train, y_train), 4))        # type: ignore

    joblib.dump(svc_grid.best_estimator_, path.join(config.models_path, 'model.pkl'), compress = 1)

    print("==> Model trained and saved on:", '\n', path.join(config.models_path, 'model.pkl'))


def metadata(**kwargs):
    """Generate metadata for model governance using testing!

    NOTE
    ----
    workspace_path: config.workspace_path

    In this section you should save your performance model,
    like metrics, maybe confusion matrix, source of the data,
    the date of training and other useful stuff.

    You can save like as workspace/performance.json:

    {
       'name': 'My Super Nifty Model',
       'metrics': {
           'accuracy': 0.99,
           'f1': 0.99,
           'recall': 0.99,
        },
       'source': 'https://archive.ics.uci.edu/ml/datasets/iris'
    }
    """
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")

    df_test = pd.read_csv(path.join(config.data_path, 'test.csv'))
    df_test.dropna(inplace=True)
    df_test = df_test.astype(str)

    X_test = df_test['review_text']
    y_test = df_test['recommend_to_a_friend']

    svc_grid = joblib.load(path.join(config.models_path, 'model.pkl'))

    print('Confusion Matrix : \n' + str(metrics.confusion_matrix(y_test, svc_grid.predict(X_test))))    # type: ignore

    print(metrics.classification_report(y_test, svc_grid.predict(X_test))) # type: ignore

    


def predict(input_data):
    """Predict: load the trained model and score input_data

    input_data: pandas.DataFrame(columns=['review_text', 'review_title'])

    NOTE
    ----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.
    """

    print("==> PREDICT DATASET {}".format(input_data))

    print(input_data)
    print(type(input_data))

    # Data check
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data, index=[0]) 

    # Data Preprocessing
    if 'review_title' in input_data.columns:
            input_data['review_text'] = input_data['review_title'] + ' ' + input_data['review_text']
            input_data.drop(columns=['review_title'], inplace=True)
    # Transform data for prediction
    input_data['review_text'] = input_data['review_text'].str.lower()
    # Merging negations
    input_data['review_text'] = input_data['review_text'].str.replace('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', 'nao', regex=True)
    # Cleaning noise
    regex_list = [r'www\S+', r'http\S+', r'@\S+', r'#\S+', r'[0-9]+', r'\W', r'\s+', r'[ \t]+$']
    for regex in regex_list:
        input_data['review_text'] = input_data['review_text'].str.replace(regex, ' ', regex=True)
    # Lemmatizer
    input_data['review_text'] = nlp_utils.lemmatizer(input_data['review_text'])
    # input_data accents
    input_data['review_text'] = input_data['review_text'].apply(lambda x: " ".join([unidecode(word) for word in x.split()]))

    svc_grid = joblib.load(path.join(config.models_path, 'model.pkl'))
    predictions = svc_grid.predict(input_data)

    print(predictions)


# Run all pipeline sequentially
def run(data_path, **kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))
    print("Running nlp_test_neoway by Gabriel Hartmann de Azeredo")
    features(data_path, **kwargs)  # generate dataset for training
    train(**kwargs)     # training model and save to filesystem
    metadata(**kwargs)  # performance report


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()

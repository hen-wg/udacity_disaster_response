# import packages
import sys
import pandas as pd
import sqlite3
from data_preprocessing import clean_and_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath) -> tuple:
    """
    Load the data from the database.

    Args:
        database_filepath (str): file path to the database

    tuple:
        X (array): array of messages, y (array): array of categories, category_names (list): list of category names

    """
    # read in file
    # connect to the database
    conn = sqlite3.connect(database_filepath)
    # conn = sqlite3.connect('../data/DisasterResponse.db')
    # run a query
    df = pd.read_sql('SELECT * FROM disaster_response_messages', conn)
    # close the connection
    conn.close()

    # define features and label arrays
    y = df[df.columns[4:]].values
    X = df['message'].values
    category_names = df.columns[4:]

    return X, y, category_names


def tokenize(text: str) -> list:
    """
    Clean and tokenize text data.

    Args:
        text (str): Text data.

    Returns:
        list: Cleaned and tokenized text data.    
    """
    cleaned_tokens = clean_and_tokenize(text)
    return cleaned_tokens


def build_model():
    """Build model pipeline.

    Returns:
        model (pipeline):  model pipeline optimized with GridSearchCV
    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3],
    }

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters, refit=True, n_jobs=-1, verbose=3, cv=2)

    # return model_pipeline
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate model performance.

    Args:
        model (pipeline): model pipeline
        X_test (array): array of test messages
        y_test (array): array of test categories
        category_names (list): list of category names
    Returns:
        Classification report for each category           
    """
    # predict on test data
    y_pred = model.predict(X_test)

    # print classification report
    for cols in category_names:
        print(f"Category: {cols}\n", classification_report(
            y_test[:, category_names.get_loc(cols)], y_pred[:, category_names.get_loc(cols)]))
    pass


def save_model(model, model_filepath):
    """
    Save the model to a pickle file.

    Args:
        model (pipeline): model pipeline to be saved
        model_filepath (str): path to save model
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# def train(X, y, model):
#     # train test split
#     # fit model
#     # output model test results
#     return model


# def export_model(model):
#     # Export model as a pickle file


# def run_pipeline(data_file):
#     X, y = load_data(data_file)  # run ETL pipeline
#     model = build_model()  # build model pipeline
#     model = train(X, y, model)  # train model pipeline
#     export_model(model)  # save model


# if __name__ == '__main__':
#     data_file = sys.argv[1]  # get filename of dataset
#     run_pipeline(data_file)  # run data pipeline

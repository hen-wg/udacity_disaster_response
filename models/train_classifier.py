# import packages
import sys
import sqlite3
import pandas as pd


def load_data(database_filepath):
    # read in file

    # clean data

    # load to database

    # define features and label arrays

    # return X, y
    pass


def tokenize(text):
    pass


def build_model():
    # text processing and model pipeline

    # define parameters for GridSearchCV

    # create gridsearch object and return as final model pipeline

    # return model_pipeline
    pass

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


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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

import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    """Load data from csv files and merge them into a single dataframe."""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, how='inner', on='id')
    return df


def clean_data(df) -> pd.DataFrame:
    """Clean the data by splitting the categories into separate columns, extracting the values from category columns and removing duplicates."""

    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # use first row to extract the column names
    row = categories.head(1)
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()[0]
    categories.columns = category_colnames

    # extract the numeric values from the category columns
    categories = categories.apply(lambda x: x.str[-1].astype(int), axis=0)

    # remove the original categories column
    df.drop('categories', axis=1, inplace=True)

    # concatenate the new categories columns with the original dataframe
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Save data to a sqlite database."""
    engine = create_engine('sqlite:///' + database_filename)
    # log the amount of rows written to database
    print(f"Saving {df.shape[0]} rows to table disaster_response_messages.")
    df.to_sql('disaster_response_messages', engine,
              index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

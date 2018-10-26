import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories data and merges them
    into one dataframe
    
    Input: Filepath to messages and categories files
    Output: Two data frames. One for each file
    """
    # Load two csv files
    messages = pd.read_csv("messages_filepath")
    categories = pd.read_csv("categories_filepath")
    
    # Merge the two files using the common id
    df = messages.merge(categories, on = 'id')
    
    return df


def clean_data(df):
    """
    Clean the datafile.
    
    Input: Merged dataframe
    Output: Cleaned dataframe
    """
    # The categories as all in one column so need to split them
    categories = df.categories.str.split(pat=";", expand=True)
    
    # Remove two digits at the end of the category names and
    # change the column headers
    row = categories.iloc[0, :].values
    category_colnames = list(map(lambda x: x[:-2], row))
    categories.columns = category_colnames
    
    # The last digit in the categories represents the
    # values for that category. Need to set them as values
    # and convert to integers
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = categories[column].apply(lambda x: int(x))
    
    # The category "related" consists of values 0, 1, and 2 but we only want
    # 0 and 1. Considered converting the 2s to 1s since 2 is next to 1 on the keyboard
    # and it might just be punching errors but will instead remove these
    # rows since we can't be sure of the reason
    categories = categories[categories.related != 2]
    
    # Drop the original category feature from main dataframe
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe and the new categories dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Remove any duplicate rows keeping the first instance of any duplicates
    df = df.drop_duplicates(keep='first')
    
    return df


def save_data(df, database_filename):
    """
    Function saves the cleaned dataframe to a sqlite database
    
    Inputs: df = dataframe we want to save to a database
            database_filename = What we want to call the database
    """
    engine = create_engine('sqlite:///'+database_filename))
    df.to_sql('DataTable', engine, if_exists='replace', index=False)
    
    pass  


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
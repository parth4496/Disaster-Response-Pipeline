"""
Course: Udacity Data Scientist Nanodegree
Project: Disaster Response Pipeline project

File: This file is used to process the data precisely perform the ETL process.

Args:
    - disaster_message.csv 
    - disaster_categories.csv
    - disaster_response.db
All these filenames will be required to put in as arguments to successfully run this file.


"""



# Import Relevant Files
import sys
import os
import sqlite3

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from IPython.display import display

# Load data using fn()

def load_data(messages_filepath, categories_filepath):
    """
    This function will automatically load data into df given that appropriate arguments are passed on
    
    Args:
        - messages_filepath = the path to the messages data (csv) file i.e. /data folder in our case
        - categories_filepath = the pass to the categories data file i.e. /data folder in our case
        
    Output:
    
    df = a data frame containing both the data merged.
    
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #messages.head()
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #categories.head()
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    df.head()
    
    return df
    
    
    


def clean_data(df):
    """
    This function will perform basic cleaning operations on the merged data we obtain from loading data in load_data()
    
    Args:
        - df: the dataframe that we want to clean
        
    Output:
        - df: a cleaned dataframe for basic cleaning procedures
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    categories.head()
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    #row
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0])
    print(category_colnames)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    
    # set each value to be the last character of the string
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    categories.head()
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)
    df.head()
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df




def save_data(df, database_filename):
    """
    This function will save the cleaned dataframe into a SQLite database.
    
    Args:
        - df: cleaned data frame which needs to be exported
        - database_filename: the name of the database that we want our data to be stored in.
        
    Output: The output from this function will generate a '.db' file by the name provided in argument, but wont return back anything.
    
    """
    
    engine = sqlite3.connect(database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')

    

def main():
    """
    This function will be responsible for running the ETL pipeline upon passing necessary arguments.
    
    """
    
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
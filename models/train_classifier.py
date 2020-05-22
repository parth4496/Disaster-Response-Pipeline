#pip install nltk
#pip install plotly

# import libraries
import pandas as pd
import os
import sys
import re
import pickle
import numpy as np
import sqlite3
pd.set_option('display.max_columns', None)


from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion # for implementing pipelines and feature Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV  # to split data into training and testing set
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import gmean

# We also need necessary NLTK packages
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#################################################################

def load_data(database_filepath):
    """Load the filepath and return the data"""
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    table_name = os.path.basename(database_filepath)
    df = pd.read_sql_table(table_name, con=engine) # is table always called this? 
    
    X = df['message']
    y = df[df.columns[4:]]
    
    category_names = y.columns
    
    for col in list(category_names):
        y[col] = pd.to_numeric(y[col])
    
    print(X.head())
    print(y.head())
    print(category_names)
    return X, y, category_names


######################################################################

def tokenize(text):
    """
    This function will perform the tokenization process
    
    Arguments:
        - text: the message which needs to be tokenized
        
    Output:
        - token_msgs = A list of tokens which are derived from the input messages
        
    """
    url_string = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Lets replace the url contents in messages with a string to reduce complexity
    
    detected_urls = re.findall(url_string, text)  # finds all the urls
    
    # Replace urls with string 'url_string'
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_string)
    
    # convert the words in text msgs into tokens
    tokens = nltk.word_tokenize(text)
    
    # Lemmatize the words to get it into root form
    lemmatized = nltk.WordNetLemmatizer()
    
    # change the format of lemmatized tokens to convert it to all lower case and strip white spaces for simplicity
    token_msgs = [lemmatized.lemmatize(x).lower().strip() for x in tokens]
    
    return token_msgs

###############################################################

# A custom transformer to extract the starting verb of a message
"""
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    The class mentioned here will extract the starting verb for a senence which will be used as an additional feature 
    for the classification model
    
    
    def starting_verb(self, text):
        sent_list = nltk.sent_tokenize(text)
        
        for sent in sent_list:
            pos_tags = nltk.pos_tag(tokenize(sent))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word =='RT':
                return True
        return False
    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        X_tag = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tag)
    
"""

###############################################################

    
def build_model():
    """
    This function will built a pipeline for ML model, the most appropriate model has been used from 'ipynb' file.
    
    Output: The output is an ML classifier model
        
    """
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf_transformer', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__max_depth': [10, None],
                  'clf__estimator__min_samples_leaf':[2, 5]}

    model_cv = GridSearchCV(pipeline, parameters)
    
    return model_cv


#######################################################################


def evaluate_model(model, X_test, y_test, category_names):
    """
    This function evaluates the model and uses the above function's output for its calculation
    
    Args:
        - model: the pipeline model which was return by buid_model and fitted on training dataset
        - X_test: testing data
        - y_test: testing labels data
        - category_names: label names
        
    Output: Prints the score label by label
    
    """
    y_pred = model.predict(X_test)
    
    #print(classification_report(y_test, y_pred, target_names=category_names))
    
    # Calculate the accuracy for each of them.
    
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))

    
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    
    

####################################################################



def save_model(model, model_filename):
    """
    This function will save our model in form of a pickle file
    
    
    Args: 
        - model: the model that is to be stored
        - model_filepath: the path where we want to store the pickle file
        
    Output:
        The output will be in the form of a pickle file at given path
   
    """
    
    with open(model_filename, 'wb') as file:  
        pickle.dump(model, file)
    
    
    
    
    
    
##################################################################




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
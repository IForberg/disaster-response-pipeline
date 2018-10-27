import sys
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])

import sys
import pandas as pd
import numpy as np
               

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import pickle

"""
This module loads clean data from a database, tokenizes the
text, creates a pipeline and trains a model. It then evaluates the
model and saves it to a picle file
"""


def load_data(database_filepath):
    """
    Loads clean data from a database and splits out the
    dependent and independent variables that will be used
    in the machine learning model
    
    Input: Filepath to a database
    Output: X, y and category_names variables    
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DataTable', con=engine)
    
    # Drop any NaNs
    df = df.dropna()
    
    # Split data into dependent and independent variables
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(y)
    
    return X, y, category_names


def tokenize(text):
     """
    Takes a text and tokenizes it by splitting
    it into separate words, lemmatizing it, converting
    to all lower case and stripping whitespace.
    
    Inputs: Text strings
    Output: tokenized text
    Credit: Function borrowed from a Udacity lesson
    """
    tokens = word_tokenize(message)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
       ('vect', CountVectorizer(tokenizer=tokenize)),
       ('tfidf', TfidfTransformer()),
       ('clf', MultiOutputClassifier(RandomForestClassifier()))
       ])
    
    parameters = {'clf__estimator__criterion': ['gini'],
                  'clf__estimator__max_leaf_nodes': [7],
                  'clf__estimator__min_samples_split': [2],
                  'clf__estimator__n_estimators': [100]


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function evalutes the model on a test set
    Inputs:
        model: The trained model
        X_test: Test features
        y_test: Test labels
        category_names: Array of category names (string)
    """
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, y_test.values, target_names=category_name))
    print("**** Accuracy scores for each category *****\n")
    # The feature child_alone was dropped so will loop over 35 features
    for i in range(35):
        print("Accuracy score for " + y_test.columns[i], accuracy_score(y_test.values[:,i],y_preds[:,i]))


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
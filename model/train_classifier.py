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
model and saves it to a picle file.

The dataset is fairly unbalanced with lots of 0s for every 1s
in most features so metrics like the f1-score is impacted
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
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build the machine learning model using Gridsearch
    Inputs: None
    Output: cv = best model found by Gridsearch given the parameters
    """
    # Make a pipeline
    pipeline = Pipeline([
       ('vect', CountVectorizer(tokenizer=tokenize)),
       ('tfidf', TfidfTransformer()),
       ('clf', MultiOutputClassifier(RandomForestClassifier()))
       ])
    
    # Parameters for use in the Gridsearch.
    # Took about 50 minutes to run in Jupyter Notebook
    # with better results than a trial that took 535 minutes!
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__criterion': ['entropy', 'gini'],
        'clf__estimator__class_weight' : ['balanced']
        
    }

    
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 1)
    
    return cv
    

def evaluate_model(model, X_test, y_test, category_names):
    """
    Function evalutes the model on a test set
    Inputs:
        model: The trained model
        X_test: Test features
        y_test: Test labels
        category_names: Array of category names (string)
    """
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, y_test.values, target_names=category_names))
    print("-" * 45)
    print("**** Accuracy scores for each category *****\n")
    # Loop over 36 features
    for i in range(36):
        print("Accuracy score for " + y_test.columns[i], accuracy_score(y_test.values[:,i],y_preds[:,i]))


def save_model(model, model_filepath):
    """
    Saves the trained model in a pickle file
    so you can use it on new data
    
    Inputs: model = Trained model
            model_filepath = Where you want to save the pickle file
    Output: Function doesn't return anything but saves model to a file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
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
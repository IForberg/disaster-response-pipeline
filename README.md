# disaster-response-pipeline
This is a Udacity project where I make make first attempt at 
natural language processing (NLP) and making an ETL pipeline
for classification of text data.

## Installing
I used the workspace provided by Udacity in this project.

## Instructions for running the pipeline and app
To load the files, create a database and create a random forest
machine learning model run the following commands in the root directory:

- python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- python models/train_classifier.py ../data/DisasterResponse.db models/classifier.pkl

The first line loads and cleans the data before saving it to an SQLite database. The second line
creates a machine learning model using grid search and saves the trained model to a pickle file.

Run the following line while in the app directorye to start the web app:
python run.py

After running the line go to http://0.0.0.0:3001/

## Files

### data directory
`process_data.py` <br>
Contains the code for loading the csv-files, cleaning and merging them and saving the cleaned file to
an SQLite database

`DisasterResponse.db` <br>
An SQLite database containing the cleaned data

`disaster_categories.csv` and `disaster_messages.csv` <br>
The source files that were loaded and cleaned for NLP processing and analysis

### model directory
`train_classifyer.py` <br>
Code that loads the data from the SQLite database, tokenizes the data, runs a grid search, trains
and evaluates a random forest model and saves the trained model to a Pickle file.

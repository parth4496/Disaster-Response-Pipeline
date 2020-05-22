# Disaster Response Pipeline Project

## Table of content
	- Abstract
    - Installation
    - Files
    - Results
    - Acknowledgements
    - Instructions

### Abstract
	This project was aimed at implementing general Data Engineering concepts like pipelining and feature union. This project also contains scripts for ETL process and ML models for classification. Lastly, it also consists of a web dashboard which is used to do the classification of disaster response messages.

### Installation

I have used Python 3, Jupyter notebook, python scripts, Sklearn's ML models, NLTK library package for NLP tasks, sklearn's metrics to score models, SQLite3 to connect python to a db, pickle file to store models, and various other python packages


### Files

This project mainly consists of 3 folders and a README.md files

1) app
	- run.py: this file is to run the web application to display results
    - tempelates (folder): containing html files to make the web page

2) data
	- disaster_messages.csv: a csv file consisting disaster messages.
    - disaster_categories.csv: a csv file consisting of all the available categories.
    - process_data.py: This python script contains ETL code to generate a data base file which will be used for ML modeling.
    - disaster_response.db: This database file will be generated upon successful execution of the process_data.py


3) models
	- train_classifier.py: A python script consists of a pipelining of ML classification models.
    - classifier.pkl: A pickle file that will be generated upon successful execution of train_classifier.py file.



### Results

- A successful execution of ETL file will generate a database file storing clean and merged data.

- A successful execution of ML Pipelining file will generate a pickle file consisting the model used to classify messages.

- The Flask app is used to create a web based visualizations.



### Acknowledgements

	Credits must be given to Udacity for providing basic code framework and FigureEight for providing the data related to disaster messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

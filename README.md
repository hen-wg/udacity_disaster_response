# Disaster Response Pipeline Project
This project forms part of the [Udacity nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025?utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_term=143524484639&utm_keyword=udacity%20data%20science_p&gclid=Cj0KCQiA5NSdBhDfARIsALzs2EAHpUX_4D3aZrBcu_PbklsCJYBWFEupJ-i6mpiKLVpCNy_7u8hDLVoaAje4EALw_wcB). 

It uses a dataset provided by [Appen](https://www.figure-eight.com/) which contains real messages sent during disaster events. A machine learning pipeline is created to clean and tokenize the data and classify it according to certain categories so that the messages can be dealt with accordingly. The project also includes a Flask web app where an input can be given and the resulting classification returned.

# Project Folders
Please see below a description of each of the folders of interest.

### App
- Flask web app for displaying calling classifier.                                               

### Data                                               
- Input data as well as data processing script.  
- Output database written to this folder.                                 

### Model                                               
- Data modelling and data pre-processing scripts.
- Output model written to this folder. 

# Getting Started
This project uses remote development through [Windows Subsystem for Linux 2](https://docs.microsoft.com/en-us/windows/wsl/install), or running natively from Windows. The environment is controlled using `venv` and `pip` for package management.

## Installation process
In order to get the project up and running, open a terminal window and run the following commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note that the folders `udacity_disaster_response`, `data`, `models`  should be added to your `PYTHONPATH` in `venv/bin/activate`.

Example code: 
```
export PYTHONPATH="path/to/udacity_disaster_response:$PYTHONPATH"
```

## Testing
Testing was beyond the scope of this tutorial but may be included for future work.

## Running Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Navigate to `app` directory and run your web app.
    `python run.py`
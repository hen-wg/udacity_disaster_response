# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


# Project Folders
Please see below a description of each of the folders of interest. The remaining folders are part of the environment setup only.

### Functional                                                   
- 
- tests: Testing functionality.                        

### Other                                               
- data:           
- notebooks: Notebooks used for exploration and testing only.                                       


# Getting Started
This project uses remote development through [Windows Subsystem for Linux 2](https://docs.microsoft.com/en-us/windows/wsl/install), or running natively from Windows. The environment is controlled using `venv` and `pip` for package management.

## Installation process
In order to get the project up and running, open a terminal window and run the following commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note that the folder `udacity_disaster_response` should be added to your `PYTHONPATH` in `venv/bin/activate`.

Example code: 
```
export PYTHONPATH="path/to/udacity_disaster_response:$PYTHONPATH"
```

## Testing
Pytest is used to run the tests. Ensure the `vscode` testing extension is linked to pytest. 

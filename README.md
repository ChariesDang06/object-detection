# object-detection

# first time set up for project
## create virtual venv environment
Syntax:
`python -m venv <environment-name>`
Example:
`python -m venv project-libs`

## activate virtual environment for terminal
Syntax:
`<environment-name>\Scripts\activate.bat`
Example:
`project-libs\Scripts\activate.bat`

## Install required Python packages from requirement.txt file
pip install -r requirements.txt

# Run project
Step 1: activate the virtual venv you created
Step 2: this is an api project so run server is the next step:
`uvicorn server:app --host 127.0.0.1 --port 8000 --reload`


# everytime you install new package for the project
remmember to run this to update the requirement.txt file
`pip freeze > requirements.txt`

# local testesting
## if you haven't create Front-end and want to understand how thigns work:
run `python test.py`

## if you want to see the offline demo of the project idea:
run `python main.py`
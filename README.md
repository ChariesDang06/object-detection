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
Step 2: type `python main.py` in the terminal

# everytime you install new package for the project
remmember to run this to update the requirement.txt file
`pip freeze > requirements.txt`
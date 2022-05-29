For Windows [my OS]:
1. Conda should be installed for managing python versions.
2. Open Conda cmd and run "conda create -n cv_env python=3.8".
3. Next navigate to folder with files using cd command and run "conda activate cv_env".
4. Run command "pip install -r requirements.txt" to install needed modules.
5. Run the code by the command "python demo.py".

For UNIX-like operating systems:
1. Pyenv should be installed for managing python versions and dependencies.
2. Navigate to the project directory by using cd command and run "pyenv virtualenv 3.8.13 cv_env".
3. Run command "pip install -r requirements.txt" to install needed modules.
4. Run the code by the command "python demo.py".

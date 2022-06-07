For Windows [my OS]:
1. Conda should be installed for managing python versions.
2. Open Conda cmd and run "conda create -n cv_env python=3.8".
3. Create folder for files and pull data from repo.
4. Next navigate to folder with files using cd command and run "conda activate cv_env".
5. Install requirements (! with build).
6. Build package with "python3 –m build" or install it by using "pip install git+https://github.com/antgoose/head_pose.git".

For UNIX-like operating systems:
1. Pyenv should be installed for managing python versions and dependencies.
2. Navigate to the project directory by using cd command and run "pyenv virtualenv 3.8.13 cv_env".
3. Install requirements (! with build).
4. Build package with "python3 –m build" or install it by using "pip install git+https://github.com/antgoose/head_pose.git".


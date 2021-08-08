import os
import sys

def update_python_path(train_py):
    with open(train_py, 'r') as file:
        data = file.readlines()

    train_path = "/".join(train_py.split("/")[:-1])
    data[1] = f'os.environ["PYTHONPATH"] = "{train_path}"\n'

    with open(train_py, 'w') as file:
        file.writelines(data)
        
update_python_path(sys.argv[1])
import os

def make_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
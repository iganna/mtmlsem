"""
This module contains
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

import os

def create_path(path_tmp):
    """
    Create directory
    :param path_tmp: path
    """
    try:
        os.mkdir(path_tmp)
    except:
        print('Directory exists')


def showl(a):
    """
    Print a list
    :param a: list
    """
    for tmp in a:
        print(tmp)


def show(mod, f=None):
    """
    Print the model
    :param mod: model
    :param f: file
    """
    if mod is None:
        print('Cannot show a model')
        return

    if f is None:
        tmp = mod.split('\n')
        for t in tmp:
            print(t)
    else:
        print(mod, file=f)


def sort_second(val):
    """
    Get the second value in the list
    :param val: list
    """
    return val[1]

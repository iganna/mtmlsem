"""
This module contains utilities
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

import os
import re

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


def check_file(f):
    """
    Checks whether the file exists
    :param f: file
    """
    if not os.path.isfile(f):
        raise ValueError(f'File {f} does not exist')


def check_dir(d):
    """
    Checks whether the directory exists
    :param d: directory
    """
    if not os.path.isdir(d):
        raise ValueError(f'File {d} does not exist')


def intersect(list1, list2):
    """
    Intersection of two lists
    """
    return [v for v in list1 if v in list2]


def diff(list1, list2):
    """
    Intersection of two lists
    """
    return [v for v in list1 if v not in list2]


def get_groups(file):
    """
    Parsing the file
    :param file:
    :return:
    """
    pattern = r'\b\w+\b'

    check_file(file)
    with open(file, 'r') as f:
        lines = f.readlines()

    types = {words[0]: words[1:len(words)]
             for words in [re.findall(pattern, line) for line in lines]}

    return types


def get_words_in_line(line):
    """
    Parsing the file
    :param file:
    :return:
    """
    pattern = r'\b[A-Za-z]\w+\b'

    words = re.findall(pattern, line)

    return words


def filter_by_pref(list, pref):
    """
    pref
    :param list:
    :param pref:
    :return:
    """
    if pref is None:
        return list

    # p = re.compile('^.+-.+-.+$')
    # l2 = [s for s in l1 if p.match(s)]

    return [s for s in list if s.startswith(pref)]


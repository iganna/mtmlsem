"""
This module contains utilities
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

import os
import re
import string
import numpy as np
import pandas as pd

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


def is_symmetric(a: np.array):
    """
    Check whether the matrix is symmetric
    :param a:
    :return:
    """
    tol = 1e-10
    return (np.abs(a - a.T) <= tol).all()


def check_names(names):
    """
    Check that all elements in a list start with a letter
    :param names:
    :return:
    """

    ALPHA = string.ascii_letters
    ALPHA += '_'
    for s in names:
        if not s.startswith(tuple(ALPHA)):
            raise ValueError(f'Variable name {s} is not supported')
        if ':' in s:
            raise ValueError('Symbol ":" is not allowed in names of snps/phenotypes')


def translate_chr(names: list[str]):
    """
    Translate chr names into chromosome numbers.

    Parameters
    ----------
    names : list[str]
        List of chromosomal names.

    Returns
    -------
    List[int] of chromosome numbers.

    """
    if type(names) is str:
        names = [names]
    ptrn = re.compile(r'\w+?(\d+)(?:\.\d+)?\s*$')
    min_chr = float('inf')
    chrs = [0] * len(names)
    for i, name in enumerate(names):
        try:
            c = int(name)
        except ValueError:
            c = ptrn.findall(name)
            if not c:
                raise NameError(f"Incorrect chromosomal name: {name}. "
                                "It should be either integer or be similar to "
                                "'CP027633.1'.")
        if c < min_chr:
            min_chr = c
        chrs[i] = c
    for i in range(len(chrs)):
        chrs[i] -= min_chr - 1
    return chrs
        
    
def translate_names(names: list[str]):
    """
    Translate snp names into pairs of chromosome numbers and positions.

    Parameters
    ----------
    names : list[str]
        List of snp names.

    Returns
    -------
    List[int] of chromosome numbers.

    """
    if type(names) is str:
        names = [names]
        single = True
    else:
        single = False
    chrs = list()
    pos = list()
    for name in names:
        t = name.split('_')
        if len(t) < 2:
            t = name.split(':')
        if len(t) < 2:
            t = name.split('.')
        if len(t) < 2:
            raise NameError(f"Incorrect SNP name: {name}.")
        c = '_'.join(t[:-1])
        try:
            p = int(t[-1])
        except ValueError:
            raise NameError(f"Can't parse SNP position in {name}.")
        chrs.append(c)
        pos.append(p)
    chrs = translate_chr(chrs)
    if single:
        return chrs[0], pos[0]
    return chrs, pos


def unique_mapping(x):
    """
    Retrieve a mapping of unique columns to duplicates.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Pandas DataFrame or numpy two-dimensional array.

    Returns
    -------
    uniques : dict
        Mapping of unique columns to its duplicates. The first occurence of 
        the column is considered to be "unique".
    """
    
    if type(x) is pd.DataFrame:
        cols = x.columns
        x = x.values
    else:
        cols = None
    s = x.mean(axis=0)
    med = np.median(x, axis=0)
    d = dict()
    uniques = dict()
    for i in range(x.shape[1]):
        tx = x[:, i]
        k = (s[i], med[i])
        if k not in d:
            d[k] = [i]
            uniques[i] = list()
        else:
            lt = d[k]
            try:
                j = next(filter(lambda y: np.all(x[:, y] == tx), lt))
                uniques[j].append(i)
            except StopIteration:
                d[k].append(i)
                uniques[i] = list()
    if cols is not None:
        uniques = {cols[i]: [cols[j] for j in lt] for i, lt in uniques.items()}
    return uniques
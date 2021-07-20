"""
This module contains functions for Horn's parallel analysis in python
"""
import numpy as np
from sklearn.decomposition import PCA
from pandas import DataFrame

def pa(x : DataFrame, n_shuffle=100):
    """
    :param x: Dataframe with variables in columns and observations in rows
    :param n_shuffle: the number of shuffles
    :return:
    """

    n_variables = x.shape[1]
    pca = PCA(n_components=n_variables)

    pca.fit(x)

    var_init = pca.explained_variance_
    var_shuffle = np.zeros((n_shuffle, n_variables))

    x_tmp = x.copy()
    for i in range(n_shuffle):
        for j in range(n_variables):
            x_tmp.iloc[:, j] = np.random.permutation(x_tmp.iloc[:, j])
        pca.fit(x_tmp)
        var_shuffle[i] = pca.explained_variance_

    var_mean = var_shuffle.mean(axis=0)

    for n_factors in range(n_variables):
        if var_mean[n_factors] > var_init[n_factors]:
            break

    return n_factors

"""
Function to construct the latent structure
"""

import warnings
from semopy.efa import explore_pine_model
from semopy import ModelEffects, Model, ModelGeneralizedEffects
from semopy.utils import calc_reduced_ml

import numpy as np
import pandas as pd
from itertools import combinations, permutations


from .dataset import Data, CVset

from .utils import showl

# Function for factor analysis
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

from .parallel_analysis import pa


def get_fa_loads(d_phens,
                 kmo_threshold=0.6,
                 bartlett_threshold=0.05,
                 n_shuffle=100,
                 test_factorability=False):
    """
    Get factors
    :param d_phens:
    :param loading_thresh:
    :param kmo_threshold:
    :param bartlett_threshold:
    :param n_shuffle:
    :param test_factorability:
    :return:
    """

    # Evaluation of the “factorability” of phenotypes
    if test_factorability:
        _, bartlett_value = calculate_bartlett_sphericity(d_phens)
        _, kmo_model = calculate_kmo(d_phens)
        if (kmo_model < kmo_threshold) or (bartlett_value > bartlett_threshold):
            # raise ValueError('Phenotypic data does not contain factors')
            warnings.warn('\nPhenotypic data does not contain factors')
            return None

    # Define the number of afctors by parallel analysis
    n_factors = pa(d_phens, n_shuffle)

    # factor analysis
    fa = FactorAnalyzer(n_factors=n_factors)
    fa.fit(d_phens)

    loads = pd.DataFrame(data=fa.loadings_, index=d_phens.columns)

    return loads


def get_factors(loads,
                loading_cutoff=0.5,
                echo=False):
    """
    Get factors from loadings
    :param loads: pandas dataframe: columns - phenotypes, rows - factors
    :param loading_cutoff:
    :param echo:
    :return:
    """

    if loads is None:
        return []

    bool_loads = abs(loads) >= loading_cutoff

    n_factors = loads.shape[1]

    phens_factors = []
    for i in range(n_factors):
        tmp = [v for (v, j) in zip(bool_loads.index, bool_loads.loc[:, i].to_numpy()) if j == True]
        if len(tmp) < 2:
            continue
        phens_factors += [tmp]

    if echo:
        showl(phens_factors)

    return phens_factors


def get_loading_cutoff(cv_data: CVset,
                       loadings_cutoffs=None,
                       echo=False):
    """
    Get loading cutoff by cross-validation
    :param data:
    :param loading_thresh:
    :param kmo_threshold:
    :param bartlett_threshold:
    :param n_shuffle:
    :param test_factorability:
    :return:
    """
    if loadings_cutoffs is None:
        loadings_cutoffs = [x / 100 for x in range(20, 99, 1)]

    # Get loadings in cross-validation
    n_cv = len(cv_data.train)
    cv_loads = []
    for d in cv_data.train:
        cv_loads += [get_fa_loads(d_phens=d.d_phens)]

    jacc_loading = []
    for cutoff in loadings_cutoffs:
        jacc_pw = []
        n_fac_pw = []
        for i, j in combinations(range(n_cv), 2):
            f1 = get_factors(cv_loads[i], cutoff)
            f2 = get_factors(cv_loads[j], cutoff)

            n_fac_pw += [len(f1), len(f2)]

            if (len(f1) < len(f2)):  # If the number of factors in one
                f1, f2 = (f2, f1)


            for i1 in range(len(f1)):
                jacc_tmp_max = 0
                for i2 in range(len(f2)):
                    phen_common = [p for p in f1[i1] if p in f2[i2]]
                    phen_all = set(f1[i1] + f2[i2])
                    jacc_tmp = len(phen_common) / len(phen_all)
                    jacc_tmp_max = max(jacc_tmp_max, jacc_tmp)
                jacc_pw += [jacc_tmp_max]

        jacc_loading += [[np.mean(jacc_pw)] + [min(n_fac_pw), max(n_fac_pw)]]
        if echo:
            print(f'cutoff: {cutoff}; jaccard: {jacc_loading[-1]}')

    n_fa_max = max(v[2] for v in jacc_loading)
    for n_fa in range(n_fa_max, 0, -1):
        j_fix = [j for j, n1, n2 in jacc_loading if n1 == n2 == n_fa]
        if len(j_fix) == 0:
            continue
        j_max = max(j_fix)
        idx_max_jaccard = [i for i, jacc in enumerate(jacc_loading)
                           if jacc[0] == j_max and jacc[1] == jacc[2] == n_fa]
        break

    # idx_max_jaccard = [i for i, val in enumerate(jacc_loading)
    #                    if val == max(jacc_loading[:, 1])]

    # print(jacc_loading)
    print(jacc_loading[min(idx_max_jaccard)])
    print(loadings_cutoffs[min(idx_max_jaccard)])

    return loadings_cutoffs[min(idx_max_jaccard)]



def get_structure_unconnect(data: Data,
                            loading_cutoff=None,
                            f_pref='F',
                            mod_pref='mod',
                            get_mod_full=False):
    """
    Return unconnected models
    :param data:
    :param loading_cutoff:
    :param f_pref:
    :return:
    """
    # Setup
    s = ' + '

    if loading_cutoff is None:
        loading_cutoff = 0.5

    # Get factors
    loads = get_fa_loads(d_phens=data.d_phens)
    phens_factors = get_factors(loads, loading_cutoff=loading_cutoff)


    mods = dict()
    mod_full = ''
    for i, tmp in enumerate(phens_factors):
        mod_tmp = f'{f_pref}{i} =~ {s.join(tmp)}'
        mod_full = f'{mod_full}\n{mod_tmp}'
        if not get_mod_full:
            mods[f'{mod_pref}{i}'] = mod_tmp
    if get_mod_full:
        mods['mod_full'] = mod_full
    return mods


def get_structure_connected(data: Data,
                            loading_cutoff=None,
                            use_kinship=True):
    mod = get_structure_unconnect(data, loading_cutoff=loading_cutoff, get_mod_full=True)['mod_full']

    # get sem model and estimate sem
    if(use_kinship):

        sem = ModelEffects(mod)
        sem.fit(data.d_all, group='group', k=data.d_kinship)
        sem_inspect = sem.inspect()
        # print(sem_inspect.loc[1:10, 'Estimate'])


        sem = ModelGeneralizedEffects(mod, effects='group')
        sem.fit(data.d_all, group='group', k=data.d_kinship)
        sem_inspect = sem.inspect()
        # print(sem_inspect.loc[1:10, 'Estimate'])
    else:
        sem = Model(mod)
        sem.fit(data.d_all)
        sem_inspect = sem.inspect()
        # print(sem_inspect.loc[1:10, 'Estimate'])

    # Fix parameters

    # add influencies from one factor to another


    lat_vars = sem.vars['latent']
    # TODO while to add more relations, use hyperparameters for stability

    ml_min = 10e10
    mod_min = mod
    for f1, f2 in permutations(lat_vars, 2):
        mod_tmp = f'{mod}\n{f1} ~ {f2}'
        sem = Model(mod_tmp, cov_diag=True)
        sem.fit(data.d_all)
        res = calc_reduced_ml(sem, data.phens)

        if ml_min > res:
            mod_min = mod_tmp

    return dict(mod_connected=mod_min)


def get_structure_picea(data: Data,
                         loading_cutoff=None,
                        f_pref='F',
                        mod_pref='mod',
                        get_mod_full=False):
    """
    This function constructs the latent structure of a Picea (spruce) form
    :param data:
    :param loading_cutoff:
    :param f_pref:
    :param mod_pref:
    :param get_mod_full:
    :return:
    """

    d_factors = pd.DataFrame(data.d_phens)
    phen_names = data.phens
    phens_factors_all = []

    while(len(phen_names) > 2):
        loads = get_fa_loads(d_phens=d_factors.loc[:,phen_names])
        if loads is None:
            break
        phens_factors = get_factors(loads, loading_cutoff=loading_cutoff)
        if len(phens_factors) == 0:
            break

        n_f = len(phens_factors_all)
        phens_factors_all += phens_factors

        factors = []
        fa = FactorAnalyzer(n_factors=1)
        for phens in phens_factors:
            fa.fit(d_factors.loc[:, phens])
            if len(factors) == 0:
                factors = fa.transform(d_factors.loc[:, phens])
            else:
                factors = np.concatenate((factors, fa.transform(d_factors.loc[:, phens])), axis=1)

        d_factors_tmp = pd.DataFrame(factors,
                                 columns=[f'{f_pref}{i+n_f}' for i in range(factors.shape[1])],
                                     index=d_factors.index)
        phen_names = list(d_factors_tmp.columns)
        d_factors = pd.concat([d_factors, d_factors_tmp], axis=1)

    # ---------------
    # Construct descriptions of model
    s = ' + '
    mods = dict()
    for i, tmp in reversed(list(enumerate(phens_factors_all))):
        s_f = f' {f_pref}{i}' # SPACE SYMBOL AT THE BEGINNING IS IMPORTANT
        for k, m in mods.items():
            if m.find(s_f) != -1:
                mods[k] += f'\n{f_pref}{i} =~ {s.join(tmp)}'
                s_f = ''
                break

        if s_f == '':
            continue

        mods[f'{mod_pref}{i}'] = f'{f_pref}{i} =~ {s.join(tmp)}'

    if get_mod_full:
        mod_full = dict(full='')
        for k, m in mods.items():
            mod_full['full'] += '\n' + m
        return mod_full

    return mods


def get_structure_optics(data_phens, std=True, **kwargs):
    """
    Retrieve latent structure of the model given observations.

    Applies semopy's iternal method using OPTICS and Sparse PCA.
    Parameters
    ----------
    data_phens : pd.DataFrame
        Pandas DataFrame.
    std : bool, optional
        If True, then data is standardized beforehand. The default is True.
    **kwargs : dict
        Extra arguments to semopy.efa.explore_pine_model.

    Returns
    -------
    semopy_desc : str
        Model description.

    """
    
    if std:
        data_phens -= data_phens.mean()
        data_phens /= data_phens.std()
    semopy_desc = explore_pine_model(data_phens, **kwargs)
    showl(semopy_desc)
    return semopy_desc


def get_structure_prior():

    pass


def inspect2mod(sem_inspect):
    mod = ''
    for _, row in sem_inspect.iterrows():
        if row['op'] == 'RF':
            continue

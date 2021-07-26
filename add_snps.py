"""
Functions to add SNPs
"""


import numpy as np
from semopy import Model as semopyModel
from semopy.utils import calc_reduced_ml
from pandas import DataFrame, concat
from dataset import Data

from func_util import *

from itertools import product


def add_snps(mod,
             data: Data,
             snp_multi_sorting=True,
             snp_pref=None):
    """

    :return: model and prior values of parameters
    """

    thresh_mlr = 0.01
    thresh_sign_snp = 0.05
    thresh_abs_param = 0.001

    sem_mod = semopyModel(mod)
    vars_ordered = sem_traversing(mod)
    vars_lat_ord = [v for v in vars_ordered
                    if v in sem_mod.vars['latent']]
    vars_phen_ord = [v for v in vars_ordered
                    if v in data.phens]

    # Estimate init model and create new model with fixed parameters
    # sem_mod.fit(concat([data.d_phens, data.d_snps], axis=1))
    sem_mod.fit(data.d_all)
    mod_init = '\n'.join(parse_descr(sem_mod=sem_mod))
    show(mod_init)

    sem_mod_init = semopyModel(mod_init)
    sem_mod_init.fit(data.d_all)

    for variable in vars_lat_ord:
        print(variable)
        print(mod_init)
        print('-----------')
        mod_init = add_snps_for_variable(mod_init, data, variable,
                              thresh_mlr=thresh_mlr,
                              thresh_sign_snp=thresh_sign_snp,
                              thresh_abs_param=thresh_abs_param,
                              snp_pref=snp_pref)
        print(mod_init)
        print('-----------')
        print('-----------')

    # form specific variables instead of latent ones

    return mod_init



def add_snps_for_variable(mod,
                          data: Data,
                          variable,
                          thresh_mlr=0.001,
                          thresh_sign_snp=0.05,
                          thresh_abs_param=0.001,
                          snp_pref=None):
    snp_skip = []
    mod_init = f'{mod}'
    for _ in range(10):
        mod_new, snp_skip = \
            one_snp_for_variable(mod_init, data, variable,
                                 snp_skip=snp_skip,
                                 thresh_mlr=thresh_mlr,
                                 thresh_sign_snp=thresh_sign_snp,
                                 thresh_abs_param=thresh_abs_param,
                                 snp_pref=snp_pref)
        if mod_new is None:
            print('NO SNPs added')
            break
        mod_init = mod_new

    return mod_init


def one_snp_for_variable(mod_init,
                         data: Data,
                         variable,
                         snp_skip,
                         thresh_mlr,
                         thresh_sign_snp,
                         thresh_abs_param,
                         snp_pref=None,):
    """
    This fucntion tests SNPs and add one SNP for a variable
    :param mod_init: model with some fixed parameters
    :param variable: a variable to add SNP for
    :param data: training dataset
    :param snp_skip: list of SNPs to skip
    :param tune: boolean flag to restrict the variance of random errors
    :param mx_cov: covariance matrix
    :return: model with the included SNP and list of SNPs to exclude in further consideration
    """

    # Initialisation
    v_tmp = 'tmp'

    # New models
    mod_tmp = f'{mod_init}\n{variable} ~ {v_tmp}'
    mod_zero = f'{mod_init}\n{variable} ~ 0*{v_tmp}'
    sem_mod_init = semopyModel(mod_init)  # without tmp dummy variable
    sem_mod_tmp = semopyModel(mod_tmp)  # with tmp variable
    sem_mod_zero = semopyModel(mod_zero)  # with tmp variable, but fixed influence to 0

    # New data
    snp_all = data.snps
    if snp_pref is not None:
        snp_all = filter_by_pref(snp_all, snp_pref)
    snp_in = intersect(snp_all, sem_mod_init.vars['all'])
    phens_in = intersect(data.phens, sem_mod_init.vars['all'])

    data_tmp = concat([data.d_phens[phens_in], data.d_snps[snp_in]], axis=1)
    data_tmp[v_tmp] = np.zeros(data.n_samples)
    snp = data.snps[0]
    data_tmp[v_tmp] = data.d_snps[snp]

    # Fit models
    sem_mod_init.fit(data_tmp)
    sem_mod_zero.fit(data_tmp)

    fit_init_reduced = calc_reduced_ml(sem_mod_zero, data.phens)
    fit_zero_reduced = calc_reduced_ml(sem_mod_zero, data.phens)

    print(fit_zero_reduced, fit_init_reduced)
    if abs(fit_zero_reduced - fit_init_reduced) > 0.01:
        raise ValueError('Something is going wring')

    # Try all SNPs

    snp_list = []
    print(f'Skip {len(snp_skip)} SNPs')

    for snp in snp_all:

        if snp in snp_skip:
            continue
        try:
            # print(snp)
            # Fit the model

            # TODO tune optimizer: what was it for?

            data_tmp[v_tmp] = data.d_snps[snp]
            sem_mod_tmp.fit(data_tmp, clean_slate=True)
            fit_tmp_reduced = calc_reduced_ml(sem_mod_tmp, data.phens)
            fit_delta = fit_init_reduced - fit_tmp_reduced
            # print(fit_delta)

            effect = [[row['Estimate'], row['p-value']] for _, row in sem_mod_tmp.inspect().iterrows()
                      if (row['lval'] == variable) and
                      (row['rval'] == v_tmp) and
                      (row['op'] == '~')]
            if len(effect) > 1:
                raise ValueError("S")
            param_val, pval = effect[0]

            snp_list += [(snp, fit_delta, pval)]

            # If the increment of MLR is small - stop considering the SNP
            if -fit_delta < thresh_mlr:
                snp_skip += [snp]
                continue

            # If the influence is not significant - stop considering the SNP
            if pval > thresh_sign_snp:
                snp_skip += [snp]
                continue

            # If the influence is not high - stop considering the SNP
            if abs(param_val) < thresh_abs_param:
                snp_skip += [snp]
                continue


        except KeyboardInterrupt:
            raise
        except:
            snp_skip += [snp]
            continue

        snp_list += [(snp, fit_delta, param_val)]


    # If no SNPs improves the model
    if len(snp_list) == 0:
        return None, snp_skip

    # print(snp_list)

    # Get the best SNP
    snp_max, snp_val, fit_delta = get_best_snp(snp_list)
    snp_skip += [snp_max]  # To remove from further consideration

    # Add SNP to the model
    mod_max = f'{mod_init}\n{variable} ~ {snp_val}*{snp_max}'
    data_tmp[snp_max] = data.d_snps[snp_max]
    sem_mod_max = semopyModel(mod_max)
    fit_max = sem_mod_max.fit(data_tmp)
    print(fit_max.fun)

    fit_anp_reduced = calc_reduced_ml(sem_mod_max, data.phens)


    showl([fit_init_reduced, fit_anp_reduced, fit_delta])

    return mod_max, snp_skip


def get_best_snp(snp_list):
    """
    This function choses the best SNP from the tested list
    :param snp_list: list of SNPs with log-likelihood values
    :return: name of the best SNP anf its loading value
    """
    # Get the best SNP
    snp_max = ''
    snp_val = 0
    delta_max = snp_list[0][1]
    for snp, delta, val in snp_list:
        if delta >= delta_max:
            delta_max = delta
            snp_max = snp
            snp_val = val
    return snp_max, snp_val, delta_max


def sem_var_order(descr):
    """
    String description of the mtmlSEM model
    :param descr: string
    :return: lists of latent and phenotype variables
    """

    descr_lines = descr.split('\n')
    sem_mod = semopyModel('\n'.join(descr_lines))
    var_lat = list(sem_mod.vars['latent'])
    var_exo = list(sem_mod.vars['exogenous'])
    var_lat_exo = intersect(var_lat, var_exo)

    var_phen = diff(sem_mod.vars['observed'], var_exo)

    var_order = []

    while len(var_lat) > 0:

        descr_lines = [line for line in descr_lines
                       if all([line.find(lat) < 0 for lat in var_lat_exo])]

        sem_mod = semopyModel('\n'.join(descr_lines))
        var_lat_new = list(sem_mod.vars['latent'])
        var_order += diff(var_lat, var_lat_new)

        var_lat = var_lat_new
        var_exo = list(sem_mod.vars['exogenous'])
        var_lat_exo = intersect(var_lat, var_exo)
    # print(var_order)

    return var_order, var_phen


def sem_traversing(descr):
    """
    String description of the mtmlSEM model
    :param descr: string
    :return: lists of latent and phenotype variables
    """

    descr_lines = parse_descr(descr)
    sem_mod = semopyModel('\n'.join(descr_lines))
    var_exo = list(sem_mod.vars['exogenous'])
    var_all = list(sem_mod.vars['all'])
    var_order = []

    while len(var_exo) > 0:
        descr_lines = [line for line in descr_lines
                       if all([line.find(lat) < 0 for lat in var_exo])]

        sem_mod = semopyModel('\n'.join(descr_lines))
        var_exo_new = list(sem_mod.vars['exogenous'])
        var_order += diff(var_exo, var_exo_new)
        var_exo = var_exo_new

    var_order += diff(var_all, var_order)
    showl(var_order)

    return var_order



def parse_descr(descr=None, sem_mod=None):
    """
    Translate the description of the model
    into interactions between pair of variables
    :param descr: String
    :return:
    """
    # print(descr, sem_mod)
    if (descr is None) and (sem_mod is None):
        raise ValueError('Please provide arguments')
    if descr is None:
        descr = sem_mod.description
        effects = sem_mod.inspect()


    descr_lines = descr.split('\n')
    lines_spart = [line for line in descr_lines
                   if (line.find('~') > 0) and (line.find('=~') < 0)]
    lines_mpart = [line for line in descr_lines
                   if line.find('=~') > 0]



    descr_parse = []
    for line in lines_spart:
        tmp = line.split('~')
        indicators = get_words_in_line(tmp[0])
        predictors = get_words_in_line(tmp[1])
        pairs = list(product(indicators, predictors))
        if sem_mod is None:
            descr_parse += [f'{p[0]} ~ {p[1]}' for p in pairs]
        else:

            vals = [effects[(effects['lval'] == p[0]) &
                             (effects['rval'] == p[1])].iloc[0]['Estimate'] for p in pairs]
            descr_parse += [f'{p[0]} ~ {v} * {p[1]}' for p, v in zip(pairs, vals)]


    for line in lines_mpart:
        tmp = line.split('=~')
        predictors = get_words_in_line(tmp[0])
        indicators = get_words_in_line(tmp[1])
        pairs = list(product(indicators, predictors))
        if sem_mod is None:
            descr_parse += [f'{p[0]} ~ {p[1]}' for p in pairs]
        else:
            vals = [effects[(effects['lval'] == p[0]) &
                            (effects['rval'] == p[1])].iloc[0]['Estimate'] for p in pairs]
            descr_parse += [f'{p[1]} =~ {v} * {p[0]}' for p, v in zip(pairs, vals)]


    # showl(descr_parse)

    return descr_parse



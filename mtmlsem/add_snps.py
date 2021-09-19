"""
Functions to add SNPs
"""


import numpy as np
from semopy import Model as semopyModel
# from semopy import ModelMeans as semopyModel
from semopy.utils import calc_reduced_ml
from pandas import DataFrame, concat
from .dataset import Data, CVset

from .utils import *

from itertools import product
from factor_analyzer import FactorAnalyzer


class Hyperparams:
    thresh_mlr = 0.1
    thresh_sign_snp = 0.05
    thresh_abs_param = 0.1


def add_snps_residuals_cv(mod,
                         data: Data,
                         thresh_mlr=Hyperparams.thresh_mlr,
                         thresh_sign_snp=Hyperparams.thresh_sign_snp,
                         thresh_abs_param=Hyperparams.thresh_abs_param,
                         snp_pref=None,
                         n_iter=10):
    n_cv = 4
    cv_data = CVset(dataset=data, n_cv=n_cv)

    thresh_mlr_var = [0.1, 0.05, 0.01]
    thresh_sign_snp_var = [0.05, 0.01]
    thresh_abs_param_var = [0.1, 0.01]

    gwas_cv = []
    snps_added_cv = []
    for thresh_mlr, thresh_sign_snp, thresh_abs_param in \
            product(*[thresh_mlr_var,
                      thresh_sign_snp_var,
                      thresh_abs_param_var]):
        print(thresh_mlr, thresh_sign_snp, thresh_abs_param)
        gwas = []
        snps_added = []
        for i_cv in range(n_cv):
            gwas_tmp, snps_added_tmp = \
                add_snps_residuals(mod=mod,
                                   data=cv_data.train[i_cv],
                                   thresh_mlr=thresh_mlr,
                                   thresh_sign_snp=thresh_sign_snp,
                                   thresh_abs_param=thresh_abs_param,
                                   snp_pref=snp_pref,
                                   n_iter=10)

            gwas += [gwas_tmp]
            snps_added += [snps_added_tmp]

        gwas_cv += [gwas]
        snps_added_cv += [snps_added]





def add_snps_residuals(mod,
                         data: Data,
                         thresh_mlr=Hyperparams.thresh_mlr,
                         thresh_sign_snp=Hyperparams.thresh_sign_snp,
                         thresh_abs_param=Hyperparams.thresh_abs_param,
                         snp_pref=None,
                         n_iter=10):

    sem_mod = semopyModel(mod)
    sem_mod.fit(data.d_all)
    relations = sem_mod.inspect()
    relations = relations.loc[relations['op'] == '~', :]
    phens = [v for v in sem_mod.vars['all'] if v in data.phens]


    vars_ordered = sem_traversing(mod)
    vars_lat_ord = list(reversed([v for v in vars_ordered
                    if v in sem_mod.vars['latent']]))

    new_var_names = []
    for f in vars_lat_ord:
        phens_f = relations.loc[relations['rval'] == f, 'lval']
        d = data.d_phens.loc[:, phens_f]

        fa = FactorAnalyzer(n_factors=1)
        fa.fit(d)
        f_val = fa.transform(d)
        f_val = f_val.transpose()[0]
        data.d_phens[f] = f_val
        new_var_names += [f]



    gwas = dict()
    snps_added = dict()
    # for variable in vars_lat_ord:
    for f in vars_lat_ord:
        print('-----------')
        mod_init = ''
        # print(variable)
        # print(mod_init)
        mod_fact, gwas[f], snps_added[f] = \
            add_snps_for_variable(mod_init, data, f,
                                      thresh_mlr=thresh_mlr,
                                      thresh_sign_snp=thresh_sign_snp,
                                      thresh_abs_param=thresh_abs_param,
                                  # n_iter=n_iter,
                                      snp_pref=snp_pref)

        sem_mod_f = semopyModel(mod_fact)
        relations_f = sem_mod_f.inspect()
        relations_f = relations_f.loc[relations_f['op'] == '~', :]

        f_val = 0
        for snp, snp_val in zip(relations_f['rval'], relations_f['Estimate']):
            f_val += data.d_snps[snp] * snp_val

        data.d_phens[f] = f_val

        print('-----------')

    return gwas, snps_added

    print(phens)
    for p in phens:
        relations_p = relations.loc[relations['lval'] == p, :]
        p_est = 0
        for var, snp_val in zip(relations_p['rval'], relations_p['Estimate']):
            p_est += data.d_all[var] * snp_val

        p_val = d.loc[:, p]
        p_res = p_val - p_est * np.dot(p_est, p_val) / np.dot(p_est, p_est)

        p_res_name = f'residual_{p}'
        data.d_phens[p_res_name] = p_res
        new_var_names += [p_res_name]

        print('-----------')
        mod_init = ''
        mod_fact, gwas[p], snps_added[p] = \
            add_snps_for_variable(mod_init, data, p_res_name,
                                      thresh_mlr=thresh_mlr,
                                      thresh_sign_snp=thresh_sign_snp,
                                      thresh_abs_param=thresh_abs_param,
                                  # n_iter=n_iter,
                                      snp_pref=snp_pref)
        print('-----------')

    data.d_phens = data.d_phens.loc[:,
                   [v for v in data.d_phens.columns
                    if v not in new_var_names]]
    return gwas, snps_added


def add_snps(mod,
             data: Data,
             thresh_mlr=Hyperparams.thresh_mlr,
             thresh_sign_snp=Hyperparams.thresh_sign_snp,
             thresh_abs_param=Hyperparams.thresh_abs_param,
             snp_pref=None,
             n_iter=10):
    """
    Add SNPs to the model
    :return: model and prior values of parameters
    """

    sem_mod = fix_variances(semopyModel(mod, cov_diag=True))
    vars_ordered = sem_traversing(mod)
    vars_lat_ord = [v for v in vars_ordered
                    if v in sem_mod.vars['latent']]
    vars_phen_ord = [v for v in vars_ordered
                    if v in data.phens]

    # Estimate init model and create new model with fixed parameters
    # sem_mod.fit(concat([data.d_phens, data.d_snps], axis=1))
    sem_mod.fit(data.d_all)
    mod_init = '\n'.join(parse_descr(sem_mod=sem_mod))
    # show(mod_init)

    sem_mod_init = fix_variances(semopyModel(mod_init, cov_diag=True))
    sem_mod_init.fit(data.d_all)

    gwas = dict()
    snps_added = dict()
    # for variable in vars_lat_ord:
    for variable in vars_lat_ord + vars_phen_ord:
        # print(variable)
        # print(mod_init)
        print('-----------')
        mod_init, gwas[variable], snps_added[variable] = add_snps_for_variable(mod_init, data, variable,
                                      thresh_mlr=thresh_mlr,
                                      thresh_sign_snp=thresh_sign_snp,
                                      thresh_abs_param=thresh_abs_param,
                                      snp_pref=snp_pref,
                                         n_iter=n_iter)
        print('-----------')
        print('-----------')

    # form specific variables instead of latent ones

    return mod_init, gwas, snps_added



def add_snps_for_variable(mod,
                          data: Data,
                          variable,
                          thresh_mlr=Hyperparams.thresh_mlr,
                          thresh_sign_snp=Hyperparams.thresh_sign_snp,
                          thresh_abs_param=Hyperparams.thresh_abs_param,
                          snp_pref=None,
                          n_iter=100):
    snp_lists = []
    snp_skip = []
    mod_init = f'{mod}'
    snps_added = []
    for _ in range(n_iter):
        show(mod_init)
        mod_new, snp_skip, snp_list = \
            one_snp_for_variable(mod_init, data, variable,
                                 snp_skip=snp_skip,
                                 thresh_mlr=thresh_mlr,
                                 thresh_sign_snp=thresh_sign_snp,
                                 thresh_abs_param=thresh_abs_param,
                                 snp_pref=snp_pref, echo=True)
        snp_lists += [snp_list]
        if mod_new is None:
            print('NO SNPs added')
            break
        mod_init = mod_new
        snps_added += [snp_skip[-1]]

    return mod_init, snp_lists, snps_added


def one_snp_for_variable(mod_init,
                         data: Data,
                         variable,
                         snp_skip,
                         thresh_mlr=Hyperparams.thresh_mlr,
                         thresh_sign_snp=Hyperparams.thresh_sign_snp,
                         thresh_abs_param=Hyperparams.thresh_abs_param,
                         snp_pref=None,
                         echo=False):
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

    empty_mod = False
    if mod_init == '':
        empty_mod = True

    # New models
    mod_tmp = f'{mod_init}\n{variable} ~ {v_tmp}'
    mod_zero = f'{mod_init}\n{variable} ~ 0*{v_tmp}'
    # sem_mod_init = fix_variances(semopyModel(mod_init, cov_diag=True))  # without tmp dummy variable
    sem_mod_tmp = fix_variances(semopyModel(mod_tmp, cov_diag=True))  # with tmp variable
    sem_mod_zero = fix_variances(semopyModel(mod_zero, cov_diag=True))  # with tmp variable, but fixed influence to 0


    # New data
    snp_all = data.snps
    if snp_pref is not None:
        snp_all = filter_by_pref(snp_all, snp_pref)
    snp_in = intersect(snp_all, sem_mod_tmp.vars['all'])
    phens_in = intersect(data.d_phens.columns, sem_mod_tmp.vars['all'])

    data_tmp = concat([data.d_phens[phens_in], data.d_snps[snp_in]], axis=1)
    data_tmp[v_tmp] = np.zeros(data.n_samples)
    snp = data.snps[0]  #'Ca1_101073'
    # snp = 'Ca3_28437425'
    data_tmp[v_tmp] = data.d_snps[snp]

    # # Fit models
    # sem_mod_init.fit(data_tmp, clean_slate=True)
    sem_mod_zero.fit(data_tmp, clean_slate=True)
    # sem_mod_tmp.fit(data_tmp, clean_slate=True)  # 11.896190990354398
    #
    # # data_tmp[v_tmp] = data.d_snps['Ca3_28437425']
    # # sem_mod_tmp.fit(data_tmp, clean_slate=True)  # 12.96532468871669
    #
    #
    # fit_init_reduced = calc_reduced_ml(sem_mod_init, phens_in)
    if empty_mod:
        fit_zero_reduced = 10 ** 10
    else:
        fit_zero_reduced = calc_reduced_ml(sem_mod_zero, phens_in)
    # fit_tmp_reduced = calc_reduced_ml(sem_mod_tmp, data.phens)
    #
    # if echo:
    #     print(fit_zero_reduced, fit_init_reduced)
    #
    # if abs(fit_zero_reduced - fit_init_reduced) > 0.01:
    #     raise ValueError('Something is going wring')

    # Try all SNPs
    if echo:
        print(f'Skip {len(snp_skip)} SNPs')
    snp_list = []
    for snp in snp_all:

        if snp in snp_skip:
            continue
        try:
            # print(snp)
            # Fit the model

            data_tmp[v_tmp] = data.d_snps[snp]

            obj = sem_mod_tmp.fit(data_tmp, clean_slate=True)

            if empty_mod:
                fit_tmp_reduced = obj.fun
            else:
                fit_tmp_reduced = calc_reduced_ml(sem_mod_tmp, phens_in)

            fit_delta = fit_zero_reduced - fit_tmp_reduced
            # print(fit_delta)

            effect = [[row['Estimate'], row['p-value']] for _, row in sem_mod_tmp.inspect().iterrows()
                      if (row['lval'] == variable) and
                      (row['rval'] == v_tmp) and
                      (row['op'] == '~')]
            if len(effect) > 1:
                raise ValueError("S")
            param_val, pval = effect[0]

            snp_list += [(snp, fit_delta, param_val, pval)]

            # If the increment of MLR is small - stop considering the SNP
            if fit_delta < thresh_mlr:
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
            snp_list += [(snp, 0, 0, 1)]
            continue


    # If no SNPs improves the model
    if len(snp_list) == 0:
        return None, snp_skip, snp_list

    # print(snp_list)

    # Get the best SNP
    snp_max, snp_val, fit_delta = get_best_snp([v for v in snp_list
                                                if v[0] not in snp_skip])
    if snp_max is None:
        return None, snp_skip, snp_list

    snp_skip += [snp_max]  # To remove from further consideration

    # Add SNP to the model
    mod_max = f'{mod_init}\n{variable} ~ {snp_val}*{snp_max}'


    if echo:
        data_tmp[snp_max] = data.d_snps[snp_max]
        sem_mod_max = fix_variances(semopyModel(mod_max, cov_diag=True))
        sem_mod_max.fit(data_tmp, clean_slate=True)
        fit_max_reduced = calc_reduced_ml(sem_mod_max, phens_in)
        print(fit_zero_reduced - fit_max_reduced - fit_delta)

    return mod_max, snp_skip, snp_list


def get_best_snp(snp_list):
    """
    This function choses the best SNP from the tested list by the max values
    :param snp_list: list of SNPs with log-likelihood values
    :return: name of the best SNP anf its loading value
    """
    if len(snp_list) == 0:
        return None, 0, 0
    # Get the best SNP
    snp_max = ''
    snp_val = 0
    delta_max = snp_list[0][1]
    for snp, delta, val, pval in snp_list:
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
    sem_mod = fix_variances(semopyModel('\n'.join(descr_lines)))
    var_lat = list(sem_mod.vars['latent'])
    var_exo = list(sem_mod.vars['exogenous'])
    var_lat_exo = intersect(var_lat, var_exo)

    var_phen = diff(sem_mod.vars['observed'], var_exo)

    var_order = []

    while len(var_lat) > 0:

        descr_lines = [line for line in descr_lines
                       if all([line.find(lat) < 0 for lat in var_lat_exo])]

        sem_mod = fix_variances(semopyModel('\n'.join(descr_lines)))
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
    sem_mod = fix_variances(semopyModel('\n'.join(descr_lines)))
    var_exo = list(sem_mod.vars['exogenous'])
    var_all = list(sem_mod.vars['all'])
    var_order = []

    while len(var_exo) > 0:
        descr_lines = [line for line in descr_lines
                       if all([line.find(lat) < 0 for lat in var_exo])]

        sem_mod = fix_variances(semopyModel('\n'.join(descr_lines)))
        var_exo_new = list(sem_mod.vars['exogenous'])
        var_order += diff(var_exo, var_exo_new)
        var_exo = var_exo_new

    var_order += diff(var_all, var_order)
    # showl(var_order)

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


def fix_variances(sem: semopyModel, var_cutoff=0.05):
    for k, v in sem.parameters.items():
        if not k.startswith('_c'):
            continue
        v.bound = (0, var_cutoff)
    return sem

"""
This module contains functions to add SNPs to the model
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"


import numpy as np
from pandas import read_csv, Series

from func_spart import create_mod_opt, n_nonsign_pvals
from func_util import show
from tune_model import tune_ordinal_covariance, tune_cov_tmp
from semopy import Model, Optimizer


def get_fix_mod(mod, data, mx_cov=None, tune=False, get_prior=False):
    """
    This function creates the string of the new model,
    when all parameters in SPart and MPart are fixed
    :param mod: string model
    :param data: training dataset
    :param mx_cov: covariance matrix
    :param tune: boolean flag to restrict the variance of random errors
    :return: model
    """

    model, opt = create_mod_opt(mod, data, tune=tune, mx_cov_ord=mx_cov)

    priors = []
    mod_new = ''
    beta_names = model.beta_names
    for p1, p2 in model.beta_params_inds:
        mod_new += '{} ~ {}*{}\n'.format(beta_names[0][p1], opt.mx_beta[p1, p2], beta_names[1][p2])
        priors += [[beta_names[0][p1], beta_names[1][p2], opt.mx_beta[p1, p2]]]

    lambda_names = model.lambda_names
    for i in range(len(lambda_names[1])):
        eq_manifest = []
        eq_manifest += ['1*{}'.format(lambda_names[0][j])
                        for j in range(len(lambda_names[0]))
                        if opt.mx_lambda[j, i] == 1]

        priors += [[eq_manifest[-1][2:], lambda_names[1][i], 1]]
        for p1, p2 in model.lambda_params_inds:
            if p2 != i:
                continue
            eq_manifest += ['{}*{}'.format(opt.mx_lambda[p1, p2], lambda_names[0][p1])]
            priors += [[lambda_names[0][p1], lambda_names[1][i], opt.mx_lambda[p1, p2]]]

        mod_new += '{} =~ {}\n'.format(lambda_names[1][i], ' + '.join(eq_manifest))


    # model_new, opt_new = create_mod_opt(mod_new, data, tune=tune, mx_cov_ord=mx_cov)
    # mlt_new = opt.optimize()
    if get_prior:
        return mod_new, priors
    return mod_new


def get_fix_mod_zero(mod, data, mx_cov):
    """
    This function creates the string of the new model,
    when all parameters in MPart are fixed;
    SPart is ZERO
    :param mod: string model
    :param data: training dataset
    :param mx_cov: covariance matrix
    :return: model
    """

    # Remove structural part
    tmp = mod.split('\n')
    tmp_zero = [t for t in tmp if ' ~ ' not in t]
    mod_zero = '\n'.join(tmp_zero)

    return get_fix_mod(mod_zero, data, mx_cov, tune=False)


def find_lat_order(model: Model):
    """
    This model defines how latent variables should be considered in DAG
    :param model: model
    :return: order of latent variables
    """
    latent_order = []

    latent_all = model.vars['Latents']
    latent_exo = model.vars['LatExo']

    latent_order += latent_exo

    beta_vars1 = model.beta_names[0]
    beta_vars2 = model.beta_names[0]

    while len(latent_order) < len(latent_all):
        for lat in latent_all:
            if lat in latent_exo:
                continue
            if lat in latent_order:
                continue

            lat_idx_in_beta = beta_vars1.index(lat)
            idx_influence = [j for i, j in model.beta_params_inds
                             if i == lat_idx_in_beta]

            var_influence = [beta_vars2[j] for j in idx_influence]

            flag = True
            for v in var_influence:
                if v not in latent_all:
                    continue
                if v not in latent_order:
                    flag = False
                    break

            if flag:
                latent_order += [lat]

    return latent_order


def mlr_degenate(model: Model, opt: Optimizer, snp_skip):
    """
    Calculate MLR without several snps
    :param model: model
    :param snp_skip: SNP names to skip
    :return:
    """

    opt.apply_degeneracy(set(snp_skip) & set(model.vars['ObsExo']))
    mlr_deg = opt.ml_wishart(opt.params)

    opt.apply_degeneracy([])
    return  mlr_deg


def mlr_wishart(m_cov, m_sigma):
    """
    F_wish = tr[S * Sigma^(-1)] + log(det(Sigma)) - log(det(S)) - (# of variables)
    We need to minimize the abs(F_wish) as it is a log of the ratio
    and the ration tends to be 1.
    """

    det_sigma = np.linalg.det(m_sigma)
    det_cov = np.linalg.det(m_cov)

    log_det_ratio = np.log(det_sigma) - np.log(det_cov)

    inv_Sigma = np.linalg.pinv(m_sigma)
    loss = np.trace(m_cov @ inv_Sigma) + log_det_ratio - m_cov.shape[0]
    return abs(loss)


def get_changed_model(mod_init, variable):
    """
    This finction changes the string representation of the model
    if one want to add SNPs to a phenotypic variable
    :param mod_init: string model
    :param variable: phenotypic trait
    :return: new model, when the trait is in the structural part
    """
    # -----------------------------------------------
    # Change the model if PC is phenotypic trait
    model_init = Model(mod_init)
    if variable not in model_init.lambda_names[0]:
        return mod_init

    mod_lines = mod_init.split('\n')
    for iline, line in enumerate(mod_lines):

        if ('=~' not in line) or (variable not in line):
            continue


        line = line.replace(' ', '')
        lat, mpart = line.split('=~')

        indicators = mpart.split('+')
        ind_remain = []
        ind_target = []
        for ind in indicators:
            tmp = ind.split('*')
            if tmp[1] == variable:
                ind_target += [ind]
            else:
                ind_remain += [ind]
        # ind_remain = [ind for ind in indicators if variable not in ind]
        # ind_target = [ind for ind in indicators if variable in ind]

        if len(ind_target) == 0:
            continue

        mpart_new = ' + '.join(ind_remain)
        mod_lines[iline] = '{} =~ {}'.format(lat, mpart_new)

        param, variable = ind_target[0].split('*')
        line_new = '{} ~ {}*{}'.format(variable, param, lat)
        mod_lines += [line_new]

        # DO NOT BREAK, because one phenotype can measure several latent factors

    mod_new = '\n'.join(mod_lines)

    return mod_new


def one_snp_for_variable_fast(mod_init, variable, data, snp_skip, tune=False, mx_cov=None):
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

    print('Skip ', len(snp_skip))
    if variable == 'FC1':
        thresh_mlr = 0.01
    else:
        thresh_mlr = 0.05
    # thresh_mlr = 0.05

    # Initialisation
    thresh_sign_snp = 0.05
    snp_all = list(data)[3:]

    # Change the model to zero-model
    mod_zero = '{}\n{} ~ 0*{}'.format(mod_init, variable, 'tmp')
    mx_cov['tmp'] = Series(np.random.rand(mx_cov.shape[0]), index=mx_cov.index)
    data['tmp'] = Series(np.random.rand(data.shape[0]), index=data.index)
    model_init, opt_init = create_mod_opt(mod_zero, data, tune=tune, mx_cov_ord=mx_cov)

    mlt_init_tmp = opt_init.optimize()
    mlt_init = mlr_degenate(model_init, opt_init, snp_skip + ['tmp'])
    print(mlt_init_tmp, mlt_init)


    mod_snp = '{}\n{} ~ {}'.format(mod_init, variable, 'tmp')
    model_snp, opt_snp = create_mod_opt(mod_snp, data, tune=tune, mx_cov_ord=mx_cov)
    params = np.array(opt_snp.params)

    snp = snp_all[0]

    # Try all SNPs
    isnp = 0
    snp_list = []
    snp = snp_all[0]


    for snp in snp_all:
        isnp += 1

        if snp in snp_skip:
            continue
        if snp[0] == 's':
            continue
        # print(snp)
        # Fit the model
        try:

            mx_cov['tmp'] = mx_cov[snp]
            data['tmp'] = data[snp]
            tune_res = tune_cov_tmp(model_snp, opt_snp, data, snp, mx_cov)
            if tune_res == False:
                snp_skip += [snp]
                # print('Tune')
                continue

            # mx_cov['tmp'] = mx_cov[snp]
            # data['tmp'] = data[snp]
            # tune_psi(model_snp, opt_snp, data, 'tmp')
            # model_snp, opt_snp = tune_ordinal_covariance(model_snp, opt_snp, mx_cov)


            # if check_mx_cov(model_snp, opt_snp) == False:
            #     snp_skip += [snp]
            #     print('False')
            #     continue

            opt_snp.params = np.array(params)
            mlt_snp = opt_snp.optimize()
            mlr_snp_deg = mlr_degenate(model_snp, opt_snp, snp_skip + ['tmp'] + [snp])
            # print(isnp, snp, mlt_snp, mlr_snp_deg)

            if np.isnan(mlt_snp) or np.isnan(mlr_snp_deg):
                snp_skip += [snp]
                # print('MLR Nan')
                continue


            # print(isnp, mlr_snp_deg)

            # If the increment of MLR is small - stop considering the SNP
            mlr_delta = mlt_init - mlr_snp_deg
            if mlr_delta < thresh_mlr:
                snp_skip += [snp]
                continue

            # If the influence is not significant - stop considering the SNP
            n_nonsign_beta, _ = n_nonsign_pvals(model_snp, opt_snp, thresh_sign_snp)
            if n_nonsign_beta > 0:
                snp_skip += [snp]
                continue

        except KeyboardInterrupt:
            raise
        except:
            snp_skip += [snp]
            continue

        if abs(opt_snp.params[0]) < 1/1000:
            snp_skip += [snp]
            continue

        snp_list += [(snp, mlr_delta, opt_snp.params[0])]

    # If no SNPs improves the model
    if len(snp_list) == 0:
        return None, None

    # print(snp_list)

    # Get the best SNP
    snp_max, snp_val = get_best_snp(snp_list)
    snp_skip += [snp_max]  # To remove from degenerate

    # Add SNP to the model
    mod_init_snp = '{}\n{} ~ {}*{}'.format(mod_init, variable, snp_val, snp_max)
    show(mod_init_snp)





    model_init, opt_init = create_mod_opt(mod_init_snp, data, tune=tune, mx_cov_ord=mx_cov)
    show(mod_init_snp)




    print(opt_init.ml_wishart(opt_init.params))
    mlt_init_tmp = opt_init.optimize()
    mlt_init = mlr_degenate(model_init, opt_init, snp_skip)
    print(mlt_init_tmp, mlt_init)
    return mod_init_snp, snp_list


    # ====================================================================================
    # ====================================================================================
    # ====================================================================================

def get_best_snp(snp_list):
    """
    This function choses the best SNP from the tested list
    :param snp_list: list of SNPs with log-likelihood values
    :return: name of the best SNP anf its loading value
    """
    # Get the best SNP
    snp_max = ''
    snp_val = 0
    mrl_max = 0
    for snp, mlr, val in snp_list:
        if mlr > mrl_max:
            mrl_max = mlr
            snp_max = snp
            snp_val = val
    return snp_max, snp_val





def snps_for_variable_fast(mod_grow, variable, data, snp_skip, tune=False, mx_cov=None):
    """
    This function adds SNPs to a variable
    :param mod_fix: model with fixed
    :param variable: a variable to add SNPs
    :param data: training dataset
    :param snp_skip: list of SNPs to skip
    :param tune: tuning parameter
    :param mx_cov: covariance matrix
    :return:
    """

    mod_init = get_changed_model(mod_grow, variable)

    mod_prev = mod_init
    n_snp = 1
    snp_add = []

    id = 0
    while True:
    # for id in range(3):
        print(id)

        # tmp = list(snp_skip)
        print(n_snp)
        n_snp += 1
        mod_snp, snp_list,  = one_snp_for_variable_fast(mod_prev, variable, data, snp_skip, tune, mx_cov)
        if mod_snp is None:
            break
        else:
            show(mod_snp)
            mod_prev = mod_snp
            snp_max, snp_val = get_best_snp(snp_list)
            snp_add += [[variable, snp_max, snp_val]]
            # snp_skip += [snp_max]

    return mod_prev, snp_add


def create_mod_from_priors(priors: str):
    """
    This function creates string model from the priors
    :param priors: list of strings, each contains tree terms:
    dependent variable, independent variable, loading
    :return: string model
    """
    latent_dict = dict()
    mod = ''
    snp_skip = []
    for line in priors.split('\n'):
        if len(line) <= 1:
            continue
        v1, v2, val = line.split('\t')
        if v1[:2] == v2[:2] == 'FC':
            if v1 not in latent_dict.keys():
                latent_dict[v1] = []
            if v2 not in latent_dict.keys():
                latent_dict[v2] = []
            mod += '\n{} ~ {}*{}'.format(v1, val, v2)

        elif v1[:2] == 'FC':
            mod += '\n{} ~ {}*{}'.format(v1, val, v2)
            snp_skip += [v2]

        else:
            latent_dict[v2] += ['{}*{}'.format(val, v1)]

    for lat in latent_dict.keys():
        tmp = ' + '.join(latent_dict[lat])
        mod += '\n{} =~ {}'.format(lat, tmp)

    return mod, snp_skip


def check_mx_cov(model_snp: Model, opt_snp: Optimizer, snp='tmp', cv_thresh = 0.9):
    """
    Check if the tmp row in Psi contains any position with a covariance higher that the threshold
    :param mx_cov: covariance matrix
    :return: boolean answer
    """
    mx_psi = opt_snp.mx_psi
    v_spart = model_snp.vars['SPart']
    idx = v_spart.index(snp)

    for i in range(mx_psi.shape[0]):
        if i == idx:
            continue
        if abs(mx_psi[i, idx]) > cv_thresh:
            return False


    return True









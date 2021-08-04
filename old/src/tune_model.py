import numpy as np
from itertools import combinations_with_replacement
from semopy import Model, Optimizer


def tune_optimizer(model: Model, opt: Optimizer):
    """
    Make Psi - diagonal
    Set variances higher than zero
    :param model: semopy model
    :param opt: semopy optimiser
    :return:
    """
    bounds = opt.bounds

    # Set Psi diagonal and Positive
    shift = model.psi_range[0]
    for i, psi in enumerate(model.parameters['Psi']):
        pos1, pos2 = psi
        if pos1 != pos2:  # Off-diagonal
            bounds[i + shift] = (0, 0)
            opt.params[i + shift] = 0
        else:  # Diagonal
            bounds[i + shift] = (0, 0.05)
            opt.params[i + shift] = max(0, min(0.05, opt.params[i + shift]))

    # Set Theta Positive
    shift = model.theta_range[0]
    for i, theta in enumerate(model.parameters['Theta']):
        pos1, pos2 = theta
        if pos1 != pos2:  # Diagonal
            bounds[i + shift] = (0, 0)
            opt.params[i + shift] = 0
        else:  # Off-diagonal
            bounds[i + shift] = (0, 0.05)
            opt.params[i + shift] = max(0, min(0.05, opt.params[i + shift]))

    opt.bounds = bounds

    tune_psi_diag(model, opt)


def tune_opt_diags(model: Model, opt: Optimizer):
    """
    Make Psi - diagonal
    Set variances higher than zero
    :param model: semopy model
    :param opt: semopy optimiser
    :return:
    """
    bounds = opt.bounds

    # Set Psi diagonal and Positive
    shift = model.psi_range[0]
    for i, psi in enumerate(model.parameters['Psi']):
        pos1, pos2 = psi
        if pos1 != pos2:  # Off-diagonal
            bounds[i + shift] = (0, 0)
            opt.params[i + shift] = 0

    # Set Theta Positive
    shift = model.theta_range[0]
    for i, theta in enumerate(model.parameters['Theta']):
        pos1, pos2 = theta
        if pos1 != pos2:  # Off-diagonal
            bounds[i + shift] = (0, 0)
            opt.params[i + shift] = 0

    opt.bounds = bounds

def tune_psi_diag(model: Model, opt: Optimizer):
    mx_psi = np.diag(np.diag(opt.mx_psi))
    opt.mx_psi = mx_psi
    model.mx_psi = mx_psi


def tune_ordinal_covariance(model: Model, opt: Optimizer, mx_cov):
    """
    Replace the covariance matrix by the predefined polychoric/polyserial
    :param model: semopy model
    :param opt: semopy optimizer
    :param mx_cov: covariance matrix
    :return:
    """

    var_all = model.vars['IndsObs']
    n_var_all = len(var_all)
    var_inds = model.vars['Indicators'] + model.vars['ObsEndo']

    mx_mod = model.mx_cov

    for i1, i2 in combinations_with_replacement(range(n_var_all), 2):
        v1 = var_all[i1]
        v2 = var_all[i2]
        if v1 in var_inds:
            if v2 in var_inds:
                mx_mod[i1, i2] = mx_mod[i2, i1] = mx_cov[v1][v2]
            else:
                mx_mod[i1, i2] = mx_mod[i2, i1] = mx_cov[v2][v1]

        else:
            if v2 in var_inds:
                mx_mod[i1, i2] = mx_mod[i2, i1] = mx_cov[v1][v2]
            else:
                if i1 == i2:
                    mx_mod[i1, i2] = 1
                # else:
                #     tmp = np.covariance(data[v1], data[v2])[0][1]
                #     mx_mod[i1, i2] = mx_mod[i2, i1] = tmp

    model.mx_cov = mx_mod
    opt.mx_cov = mx_mod
    opt.mx_cov_inv = np.linalg.inv(mx_mod)
    _, opt.cov_logdet = np.linalg.slogdet(mx_mod)


    return model, opt


# def tune_psi(model: Model, opt: Optimizer, data, snp):
#     """
#
#     :param model:
#     :param opt:
#     :param data:
#     :return:
#     """
#     var_sprat = model.vars['SPart']
#     var_lat = model.vars['Latents']
#
#     idx_psi2 = var_sprat.index(snp)
#
#     for v1 in var_sprat:
#         if v1 in var_lat:
#             continue
#
#         if v1 == snp:
#             opt.mx_psi[idx_psi2, idx_psi2] = 1
#             continue
#
#         idx_psi1 = var_sprat.index(v1)
#
#         tmp = np.cov(data[snp], data[v1])[0][1]
#         if abs(tmp) > 0.9:
#             return False
#         opt.mx_psi[idx_psi1, idx_psi2] = opt.mx_psi[idx_psi2, idx_psi1] = tmp
#
#     model.mx_psi = opt.mx_psi
#
#     return True
#     # model.load_dataset(data)
#     # opt.mx_psi = model.mx_psi


def tune_cov_tmp(model_snp: Model, opt_snp:Optimizer, data, snp, mx_cov, thresh_cov=0.9):
    var_all = model_snp.vars['IndsObs']
    n_var_all = len(var_all)
    var_inds = model_snp.vars['Indicators'] + model_snp.vars['ObsEndo']

    mx_mod = model_snp.mx_cov
    isnp = var_all.index('tmp')

    for i1 in range(n_var_all):
        if i1 == isnp:
            mx_mod[isnp, isnp] = 1
            continue
        v1 = var_all[i1]
        if v1 in var_inds:
            mx_mod[isnp, i1] = mx_cov.loc[v1, snp]
            mx_mod[i1, isnp] = mx_cov.loc[v1, snp]
            continue

        # ------------------------------------------
        # Remove NAs before calculating correlations
        idx = set(np.where(data[snp].isna())[0]) | set(np.where(data[v1].isna())[0])
        idx = list(set(range(len(data[snp]))) - idx)

        tmp = np.cov(data[snp][idx], data[v1][idx])[0][1]
        if abs(tmp) > thresh_cov:
            return False
        mx_mod[isnp, i1] = tmp
        mx_mod[i1, isnp] = tmp

    model_snp.mx_cov = mx_mod
    opt_snp.mx_cov = mx_mod
    opt_snp.mx_cov_inv = np.linalg.inv(mx_mod)
    _, opt_snp.cov_logdet = np.linalg.slogdet(mx_mod)

    # Psi matrix
    var_psi = model_snp.vars['SPart']
    idx_tmp = var_psi.index('tmp')
    opt_snp.mx_psi[idx_tmp, idx_tmp] = 1
    model_snp.mx_psi[idx_tmp, idx_tmp] = 1



    return True

"""
This module contain functions to connect laten variables (factor) into the model
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

import numpy as np

from functools import reduce
from itertools import combinations, product
from pandas import read_csv

from semopy import Model, Optimizer
from semopy.stats import calculate_p_values as calc_pvals

from tune_model import tune_optimizer, tune_opt_diags, tune_ordinal_covariance
from utils import showl, show




def no_cycle(indiv, names_latent, name_idx1, name_idx2):
    """
    This function checks whether an individual has no cycles
    :param indiv: vector of 0 and 1 - an individual
    :param names_latent: names on latent factors
    :param name_idx1: first latent factors on edges
    :param name_idx2: second latent factors on edges
    :return: True - if no cycle
             False - if cycles
    """

    n = len(names_latent)
    mx = np.zeros((n, n))
    for i in range(len(indiv)):
        if indiv[i] == 0:
            continue
        if indiv[i] == -1:
            ind1 = name_idx1[i]
            ind2 = name_idx2[i]
        else:
            ind1 = name_idx2[i]
            ind2 = name_idx1[i]

        mx[ind1, ind2] = 1

    for _ in range(n+1):
        mx = mx @ mx + mx
        mx = (mx > 0) * 1

    if sum(np.diag(mx)) != 0:
        return False
    else:
        return True


def get_model(indiv, mod_init, name_edges_there, name_edges_back):
    """
    Create model based on the individual
    :param indiv: vector of 0 and 1 - an individual
    :param mod_init: the initial measurement part
    :param name_edges_there: edges on one direction
    :param name_edges_back: edges of reverse directions
    :return: the model
    """
    mod_new = mod_init
    for i, val in enumerate(indiv):
        if val == 0:
            continue
        if val == 1:
            mod_new += '\n' + name_edges_there[i]
        else:
            mod_new += '\n' + name_edges_back[i]

    return mod_new

def prepare_data(data):
    """
    Normalize data
    :param data:
    :return:
    """
    for i in range(data.shape[1]):
        data.iloc[:, i] -= np.mean(data.iloc[:, i])
        data.iloc[:, i] /= np.std(data.iloc[:, i])

    return data


def create_mod_opt(mod_new, data, var_ordinal=None, tune=False, mx_cov_ord=None):
    """
    Create model and optimiser
    :param mod_new: the string specification of the model
    :param data: the dataset
    :return: both model and optimizer
    """
    model = Model(mod_new)
    # data = prepare_data(data)

    # var_ordinal = None
    if var_ordinal is None:
        model.load_dataset(data)
    else:
        var_ord = set(var_ordinal) & set(model.vars['Indicators'])
        model.load_dataset(data, ordcor=var_ord)
    opt = Optimizer(model)

    if tune:
        tune_optimizer(model, opt)
    else:
        tune_opt_diags(model, opt)


    if mx_cov_ord is not None:
        model, opt = tune_ordinal_covariance(model, opt, mx_cov_ord)


    return model, opt


def pvals_beta_lambda(model: Model, opt: Optimizer):
    """
    p-values for beta and lambda
    :param model:
    :param opt:
    :return:
    """
    pvals = calc_pvals(opt)
    a, b = model.beta_range
    p_beta = [pvals[i] for i in range(a, b)]
    a, b = model.lambda_range
    p_lambda = [pvals[i] for i in range(a, b)]
    return p_beta, p_lambda


def n_nonsign_pvals(model: Model, opt: Optimizer, sign_thresh):
    """
    number of non-significant parameters in beta and lambda
    :param sign_thresh:
    :param p_beta:
    :param p_lambda:
    :return:
    """
    p_beta, p_lambda = pvals_beta_lambda(model, opt)
    n_nonsign_beta = reduce(lambda x, y: x + (y > sign_thresh), [0] + p_beta)
    n_nonsign_lambda = reduce(lambda x, y: x + (y > sign_thresh), [0] + p_lambda)
    return n_nonsign_beta, n_nonsign_lambda


def grow_pipeline(path_mod, file_mod, path_data, file_data, path_res, idata=0, tune=True, mx_cov=None):
    """
    Grow-model-pipeline
    :return: final model as string
    """
    with open(path_mod + file_mod, 'r') as f:
        mod_init = f.read()
    model = Model(mod_init, )

    names_latent = model.vars['LatExo']
    n_latent = len(names_latent)

    name_edges = [[var1, var2] for var1, var2 in combinations(names_latent, 2)]
    name_idx1 = [names_latent.index(var1) for var1, _ in name_edges]
    name_idx2 = [names_latent.index(var2) for _, var2 in name_edges]
    name_edges_there = [var1 + ' ~ ' + var2 for var1, var2 in name_edges]
    name_edges_back = [var2 + ' ~ ' + var1 for var1, var2 in name_edges]
    n_edge = len(name_edges)

    sign_thresh = 0.05

    # --------------------------------------------------------------------------
    # Growth
    # --------------------------------------------------------------------------

    # the dataset
    data_file = path_data + file_data
    data = read_csv(data_file, sep='\t', header=0, index_col=0)
    if data.shape[1] == 0:
        data = read_csv(data_file, sep=',', header=0, index_col=0)

    # initla indisivuals and initial possible non-significant lambdas
    indiv_init = np.zeros(n_edge)
    lambda_thresh = 0

    indiv_res = []
    mlr_res = []
    for inewparam in range(30):

        # initial values to fund the best interaction
        new_val = (-1, -1)
        mlr_init = np.inf

        print('iter', inewparam)
        for k, val in product(range(n_edge), [-1, 1]):

            if (indiv_init[k] != 0):  # If this edge was already taken
                continue

            # Create new individual
            indiv_new = list(indiv_init)
            indiv_new[k] = val

            # check for cycles
            if no_cycle(indiv_new, names_latent, name_idx1, name_idx2) == False:
                continue

            # Optimise model
            mod_new = get_model(indiv_new, mod_init, name_edges_there, name_edges_back)
            model, opt = create_mod_opt(mod_new, data, tune=tune, mx_cov_ord=mx_cov)
            mrl_opt = opt.optimize()
            print(k, val, mrl_opt)

            # Analyse p-values
            n_nonsign_beta, n_nonsign_lambda = n_nonsign_pvals(model, opt, sign_thresh)
            # print(n_nonsign_beta, n_nonsign_lambda)
            if (n_nonsign_beta > 0) or (n_nonsign_lambda > lambda_thresh):
                mrl_sum = -np.inf
                continue


            if mrl_opt < mlr_init:
                mlr_init = mrl_opt
                new_val = (k, val)
                print('=======', mrl_opt)
            else:
                print('==old==', mlr_init, mrl_opt)

        if new_val[0] == -1:
            # lambda_thresh += 1
            continue
        else:
            # lambda_thresh = 0
            k, val = new_val
            indiv_init[k] = val

            indiv_res += [list(indiv_init)]
            # if mlr_init in mlr_res:
            #     mlr_res += [(mlr_init)]
            #     break
            mlr_res += [(mlr_init)]


    showl(mlr_res)
    idx = -1
    for i in range(1,len(mlr_res)):
        if mlr_res[i] in mlr_res[:i]:
            idx = i
            break
    print(idx)

    np.savetxt(path_res + "indivs_0" + str(idata) + ".txt",
               indiv_res[:idx], fmt='%d')
    np.savetxt(path_res + "mlr_0" + str(idata) + ".txt",
               mlr_res[:idx], fmt='%f')

    # --------------------------------------------------------------------------
    # Show the best model
    # --------------------------------------------------------------------------

    mlr_res = np.loadtxt(path_res + "mlr_0" + str(idata) + ".txt")
    indiv_res = np.loadtxt(path_res + "indivs_0" + str(idata) + ".txt")

    idx = np.argmin(mlr_res)
    indiv = indiv_res[idx]

    mod = get_model(indiv, mod_init, name_edges_there, name_edges_back)
    # show(mod)

    return mod


def grow_2layer_mod(path_mod, file_mod, path_data, file_data, path_res, mx_cov=None):
    with open(path_mod + file_mod, 'r') as f:
        mod_init = f.read()
    model = Model(mod_init, )

    names_latent = model.vars['LatExo']
    n_latent = len(names_latent)

    name_edges = [[var1, var2] for var1, var2 in combinations(names_latent, 2)]
    name_idx1 = [names_latent.index(var1) for var1, _ in name_edges]
    name_idx2 = [names_latent.index(var2) for _, var2 in name_edges]
    name_edges_there = [var1 + ' ~ ' + var2 for var1, var2 in name_edges]
    name_edges_back = [var2 + ' ~ ' + var1 for var1, var2 in name_edges]
    n_edge = len(name_edges)

    sign_thresh = 0.05

    # --------------------------------------------------------------------------
    # Growth
    # --------------------------------------------------------------------------

    # the dataset
    data_file = path_data + file_data
    data = read_csv(data_file, sep='\t', header=0, index_col=0)
    if data.shape[1] == 0:
        data = read_csv(data_file, sep=',', header=0, index_col=0)

    # initla indisivuals and initial possible non-significant lambdas
    indiv_init = np.zeros(n_edge)
    lambda_thresh = 0

    indiv_res = []
    mlr_res = []

    model, opt = create_mod_opt_2layer(mod_init, data, mx_cov_ord=mx_cov)

    mrl_opt = opt.optimize()


    pvals = calc_pvals(opt)
    a, b = model.psi_range
    p_psi = [pvals[i] for i in range(a, b)]


def create_mod_opt_2layer(mod_new, data, var_ordinal=None, mx_cov_ord=None):
    """
    Create model and optimiser
    :param mod_new: the string specification of the model
    :param data: the dataset
    :return: both model and optimizer
    """
    model = Model(mod_new)
    # data = prepare_data(data)

    # var_ordinal = None
    if var_ordinal is None:
        model.load_dataset(data)
    else:
        var_ord = set(var_ordinal) & set(model.vars['Indicators'])
        model.load_dataset(data, ordcor=var_ord)
    opt = Optimizer(model)


    if mx_cov_ord is not None:
        model, opt = tune_ordinal_covariance(model, opt, mx_cov_ord)


    return model, opt




def grow_mod_brutforce(path_mod, file_mod, path_data, file_data, mx_cov=None):
    with open(path_mod + file_mod, 'r') as f:
        mod_init = f.read()
    show(mod_init)
    model = Model(mod_init, )

    names_latent = model.vars['LatExo']
    n_latent = len(names_latent)

    name_edges = [[var1, var2] for var1, var2 in combinations(names_latent, 2)]
    name_idx1 = [names_latent.index(var1) for var1, _ in name_edges]
    name_idx2 = [names_latent.index(var2) for _, var2 in name_edges]
    name_edges_there = [var1 + ' ~ ' + var2 for var1, var2 in name_edges]
    name_edges_back = [var2 + ' ~ ' + var1 for var1, var2 in name_edges]
    n_edge = len(name_edges)


    # --------------------------------------------------------------------------
    # Get positions in Psi with significant covariances
    # We need to sort out only these interactions,
    # where Psi is significantly different from zero
    # --------------------------------------------------------------------------

    sign_thresh = 0.05

    # the dataset
    data_file = path_data + file_data
    data = read_csv(data_file, sep='\t', header=0, index_col=0)
    if data.shape[1] == 0:
        data = read_csv(data_file, sep=',', header=0, index_col=0)


    model = Model(mod_init)
    model.load_dataset(data)
    opt = Optimizer(model)


    # model, opt = create_mod_opt_2layer(mod_init, data, mx_cov_ord=mx_cov)

    mrl_opt = opt.optimize()


    pvals = calc_pvals(opt)
    a, b = model.psi_range
    p_psi = [pvals[i] for i in range(a, b)]
    psi_names = model.psi_names[0]
    beta_pos = []

    showl(list(zip(model.psi_params_inds, p_psi)))

    for i, pos in enumerate(model.psi_params_inds):
        if pos[0] == pos[1]:
            continue
        if p_psi[i] > sign_thresh:
            continue
        # We found significant interaction
        beta_idxs = [names_latent.index(psi_names[pos[0]]), names_latent.index(psi_names[pos[1]])]
        beta_idxs.sort()
        idx_tmp = [i for i in range(n_edge)
                   if name_idx1[i] == beta_idxs[0] and
                   name_idx2[i] == beta_idxs[1]]
        beta_pos += idx_tmp
    print(beta_pos)


    # --------------------------------------------------------------------------
    # Brut force of all models with betas in Psi positions
    # --------------------------------------------------------------------------

    # Construct all possible individuals
    indiv_res = np.zeros((3 ** len(beta_pos), n_edge))
    for i_indiv, indiv_combo in enumerate(product([-1, 1, 0], repeat=len(beta_pos))):
        for ipos in range(len(beta_pos)):
            indiv_res[i_indiv, beta_pos[ipos]] = indiv_combo[ipos]

    # Run the optimisation for each individual
    mlr_res = np.full(len(indiv_res), np.nan)
    for i_indiv, indiv_new in enumerate(indiv_res):

        print('---')
        if sum(indiv_new != 0) > 6:
            continue

        # Check for cycles
        if no_cycle(indiv_new, names_latent, name_idx1, name_idx2) == False:
            continue

        # Optimise model
        mod_new = get_model(indiv_new, mod_init, name_edges_there, name_edges_back)
        # model, opt = create_mod_opt(mod_new, data, tune=True, mx_cov_ord=mx_cov)
        show(mod_new)
        model = Model(mod_new, outputs_covary=False)
        model.load_dataset(data)

        # opt = Optimizer(model)

        mrl_opt = opt.optimize()

        # Get significanve
        n_nonsign_beta, n_nonsign_lambda = n_nonsign_pvals(model, opt, sign_thresh)
        if (n_nonsign_beta > 0) or (n_nonsign_lambda > 0):
            continue


        # fits = gather_statistics(opt)
        # mlr_res[i_indiv] = -fits.gfi

        mlr_res[i_indiv] = mrl_opt


        print(mrl_opt)

    # showl(list(zip(indiv_res, mlr_res)))

    # Remove NaN
    mlr_res[np.isnan(mlr_res)] = max(mlr_res[np.isnan(mlr_res) == False])

    # Get min MLR
    indiv_best = indiv_res[np.argmin(mlr_res)]
    print(indiv_best)

    # --------------------------------------------------------------------------
    # Show the best model
    # --------------------------------------------------------------------------


    mod = get_model(indiv_best, mod_init, name_edges_there, name_edges_back)
    show(mod)

    return mod


def compare_models(path_mod1, file_mod1, path_mod2, file_mod2):

    def get_phens(model1, fc):
        phen_name = model1.lambda_names[0]
        fc_name = model1.lambda_names[1]
        lambda_col = model1.mx_lambda[:, fc_name.index(fc)]
        phen_fc = [phen_name[i] for i in range(len(phen_name)) if lambda_col[i] != 0]
        return phen_fc



    with open(path_mod1 + file_mod1, 'r') as f:
        mod_init1 = f.read()
    model1 = Model(mod_init1)
    opt1 = Optimizer(model1)

    with open(path_mod2 + file_mod2, 'r') as f:
        mod_init2 = f.read()
    model2 = Model(mod_init2)


    # Find the correspondence between two models in names of latent variables
    lat1 = model1.vars['Latents']
    lat2 = model2.vars['Latents']

    idx_corresp = []

    for i, fc in enumerate(lat1):
        phens_fc = get_phens(model1, fc)
        n_common = 0
        eta_common = -1
        for j, eta in enumerate(lat2):
            phens_eta = get_phens(model2, eta)
            if len(set(phens_fc) & set(phens_eta)) > n_common:
                n_common = len(set(phens_fc) & set(phens_eta))
                eta_common = eta
        idx_corresp += [eta_common]
        
    # Find the correspondence in Beta interactions in terms of eta
    interactions1 = []
    for pos1, pos2 in model1.parameters['Beta']:
        interactions1 += [[idx_corresp[pos1], idx_corresp[pos2]]]

    interactions2 = []
    for pos1, pos2 in model2.parameters['Beta']:
        if pos2 >= len(lat2):
            continue
        interactions2 += [[model2.beta_names[0][pos1], model2.beta_names[1][pos2]]]

    n_common = 0
    n_inverse = 0
    n_redundant = 0
    n_lost = 0
    for p in interactions1:
        if p in interactions2:
            n_common += 1
            continue

        if [p[1], p[0]] in interactions2:
            n_inverse += 1
            continue

        n_redundant += 1

    n_lost = len(interactions2) - n_common - n_inverse

    print(n_common, n_inverse, n_redundant, n_lost)




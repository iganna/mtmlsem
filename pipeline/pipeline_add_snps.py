"""
This module contain the pipeline to add SNPs to the model
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

from pandas import read_csv
from multiprocess import Pool

from func_util import create_path, show
from func_spart import create_mod_opt
from func_add_snps import find_lat_order, get_fix_mod, snps_for_variable, snps_for_variable_fast


def work_with_one_model(idata, thresh):
    """

    :param idata:
    :param thresh:
    :return:
    """

    print(idata, thresh)
    tune = True

    # --------------------------------------------------------
    # Tune
    # --------------------------------------------------------

    zero_flag = False
    phen_flag = False

    path_priors = '../data/cv_model/priors_full/'

    # --------------------------------------------------------
    create_path(path_priors)



    path_cv = '../data/cv_model/'
    path_data = '../data/cv_model/norm/'

    if zero_flag:
        path_mod = '../data/cv_model/models/'
        file_mod = 'mod_' + str(thresh) + '_' + str(idata) + '.txt'
    else:
        path_mod = '../data/cv_model/mod_grow/'
        file_mod = 'mod_spart_' + str(thresh) + '_' + str(idata) + '.txt'


    file_data = 'data_train_' + str(idata) + '.txt'
    file_cov = 'cov_mx_all_' + str(idata) + '.txt'
    file_prior = 'prior_' + str(thresh) + '_' + str(idata) + '.txt'



    with open(path_mod + file_mod, 'r') as f:
        mod = f.read()

    data = read_csv(path_data + file_data, sep='\t', header=0, index_col=0)
    mx_cov = read_csv(path_cv + file_cov, sep='\t', header=0, index_col=0)
    # mx_cov=None
    model, opt = create_mod_opt(mod, data, tune=tune, mx_cov_ord=mx_cov)

    # # Fix parameters in the model
    mod_fix, priors = get_fix_mod(mod, data, tune=tune, get_prior=True, mx_cov=mx_cov)

    # Get phenotypes and letents in the correct order
    latents = find_lat_order(model)
    phens = model.vars['Indicators']
    print(phens)

    mod_grow = mod_fix
    snp_skip = []
    snp_add_all = []
    #

    mod_init = mod_grow

    for lat in latents:
        print(lat)
        show(mod_grow)
        mod_grow, snp_add = snps_for_variable_fast(mod_grow, lat, data, list(snp_skip), tune=tune, mx_cov=mx_cov)
        show(mod_grow)
        snp_add_all += snp_add
        snp_skip += [snp for _, snp, _ in snp_add]

    if phen_flag:
        mods_phen = []
        for phen in phens:
            print(phen)
            mod_tmp, snp_add = snps_for_variable_fast(mod_grow, phen, data, list(snp_skip), tune=tune, mx_cov=mx_cov)
            snp_add_all += snp_add
            mods_phen += [mod_tmp]
            snp_skip += [snp for _, snp, _ in snp_add]

    priors += snp_add_all

    with open(path_priors + file_prior, 'a') as f:
        for p in priors:
            p[2] = str(p[2])
            f.write('\t'.join(p) + '\n')


def wort_with_dataset(idata):
    for thresh in [5]:
        work_with_one_model(idata, thresh)

n_thr = 1
n_cv = 20
with Pool(n_thr) as workers:
    pmap = workers.map
    pmap(wort_with_dataset, range(2,3))


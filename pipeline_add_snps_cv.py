
import os
from mtml_model import mtmlModel
from dataset import Data, CVset
from pandas import read_csv
from optimisation import *
from semopy.efa import explore_cfa_model
from utils import *
from add_snps import *

from multiprocess import Pool


import pickle



# path_data = 'data_cicer/'
# file_phens = path_data + 'data_phens.txt'
# file_snps = path_data + 'snp_2579_renamed.txt'
# snp_pref = 'Ca'


path_data = 'data_soy/'
file_phens = path_data + 'soy_phens.txt'
file_snps = path_data + 'soy_snps.txt'
snp_pref = None
#
#
# path_data = 'data_vigna/'
# file_phens = path_data + 'vigna_phens.txt'
# file_snps = path_data + 'vigna_snps.txt'
# snp_pref = None

# path_data = 'data_flax/'
# file_phens = path_data + 'flax_phens_all.txt'
# file_snps = path_data + 'flax_snps_maf005_tab.txt'
# snp_pref = None



data_phens = read_csv(file_phens, sep='\t', index_col=0)

data_phens.isna().sum(axis=0)

# ---------------------
# # for vigna
# data_phens = data_phens.iloc[:, ['18_indiv' in s for s in data_phens.columns.to_list() ]]
# ---------------------
# # for soy
# data_phens = data_phens.iloc[:, ['Yield' not in s for s in data_phens.columns.to_list() ]]
# data_phens = data_phens.iloc[:, ['PodShat' not in s for s in data_phens.columns.to_list() ]]
# # data_phens = data_phens.iloc[:, ['Height' not in s for s in data_phens.columns.to_list() ]]
# data_phens = data_phens.iloc[:, ['Oil' not in s for s in data_phens.columns.to_list() ]]
# data_phens = data_phens.iloc[:, ['Protein' not in s for s in data_phens.columns.to_list() ]]
# # data_phens = data_phens.iloc[:, ['ilr' not in s for s in data_phens.columns.to_list() ]]

# ---------------------
# for soy paper
data_phens = data_phens.loc[:, ['Productivity', 'ilr1', 'ilr2']]
# ---------------------
# # for flax
# data_phens = data_phens.loc[data_phens['Batch'] == 'Big', :]
# # data_phens = data_phens.iloc[:, ['y2020_1' in s for s in data_phens.columns.to_list() ]]
# data_phens = data_phens.iloc[:, ['y2020_2' in s for s in data_phens.columns.to_list() ]]
# # data_phens = data_phens.iloc[:, ['y2019' in s for s in data_phens.columns.to_list() ]]
# data_phens = data_phens.iloc[:, ['FusoriumPerc' not in s for s in data_phens.columns.to_list() ]]
# data_phens.isna().sum(axis=0)
# ---------------------


data_phens = data_phens.loc[data_phens.isna().sum(axis=1) == 0, :]
data_phens = data_phens.iloc[:, list(data_phens.nunique() > 2)]
data_snps = read_csv(file_snps, sep='\t', index_col=0)



data = Data(d_snps=data_snps, d_phens=data_phens)
print(data.n_samples)
self = data

model = mtmlModel(data=data)
# model.get_lat_struct(cv=True, echo=True)
# model.show_mod()

model.get_lat_struct()
model.show_mod()

self = model
n_cv = 4


# model.add_snps(snp_pref=snp_pref)


# mod = model.mods['mod0']
# variable = 'F0'
# show(mod)
#
#
thresh_mlr = 0.1
thresh_sign_snp = 0.05
thresh_abs_param = 0.1
mod = model.mods['mod0']

n_cv = 4
cv_data = CVset(dataset=data, n_cv=n_cv)

thresh_mlr_var = [0.1, 0.05, 0.01]
thresh_sign_snp_var = [0.05, 0.01]
thresh_abs_param_var = [0.1, 0.01]


thresh_variants = list(product(*[thresh_mlr_var,
                  thresh_sign_snp_var,
                  thresh_abs_param_var]))

def func(i_variant):
    thresh_mlr, thresh_sign_snp, thresh_abs_param = thresh_variants[i_variant]
    gwas = []
    snps_added = []

    # return gwas, snps_added

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

    return gwas, snps_added
#
# gwas_cv = []
# snps_added_cv = []
# for thresh_mlr, thresh_sign_snp, thresh_abs_param in \
#         thresh_variants:
#     print(thresh_mlr, thresh_sign_snp, thresh_abs_param)
#     gwas = []
#     snps_added = []
#     for i_cv in range(n_cv):
#         gwas_tmp, snps_added_tmp = \
#             add_snps_residuals(mod=mod,
#                                data=cv_data.train[i_cv],
#                                thresh_mlr=thresh_mlr,
#                                thresh_sign_snp=thresh_sign_snp,
#                                thresh_abs_param=thresh_abs_param,
#                                snp_pref=snp_pref,
#                                n_iter=10)
#
#         gwas += [gwas_tmp]
#         snps_added += [snps_added_tmp]
#
#     gwas_cv += [gwas]
#     snps_added_cv += [snps_added]



n_thr = 2
with Pool(n_thr) as workers:
    pmap = workers.map
    res = pmap(func, range(2))

with open('res.obj', 'wb') as file:
    pickle.dump(res, file)

with open('res.obj', 'rb') as file:
    res1 = pickle.load(file)


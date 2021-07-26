
import os
from mtml_model import mtmlModel
from dataset import Data, CVset
from pandas import read_csv
from optimisation import *
from semopy.efa import explore_cfa_model
from func_util import *

path_data = 'data_cicer/'
file_phens = path_data + 'data_phens.txt'
file_snps = path_data + 'snp_2579_renamed.txt'
snp_pref = 'Ca'


# path_data = 'data_soy/'
# file_phens = path_data + 'soy_phens.txt'
# file_snps = path_data + 'soy_snps.txt'
# snp_pref = None
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

model = mtmlModel(data=data)
model.get_lat_struct(cv=True, echo=True)
model.show_mod()

model.get_lat_struct()
model.show_mod()


semopy_descr = explore_cfa_model(data_phens)
show(semopy_descr)

semopy_descr = explore_cfa_model(data_phens, mode='optics')
show(semopy_descr)


mod = model.mods['mod0']
variable = 'F0'
show(mod)


thresh_mlr = 0.01
thresh_sign_snp = 0.05
thresh_abs_param = 0.001

model = model.add_snps(snp_pref=snp_pref)

#
# descr = """ FC1 ~ FC2
# FC4 ~ FC3
# FC1 ~ FC4
# FC5 ~ FC1 + FC2
# """

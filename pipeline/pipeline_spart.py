"""
The pipeline connects factor into the structure
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

import os
from pandas import read_csv

from func_spart import grow_pipeline
from func_util import show

# ==============================================================================

var_ordinal = ['FloCol', 'StemCol', 'FlowStemCol', 'SeedCol', 'PodShape', 'SeedShape',
               'BushShape', 'StemBr', 'StemBr1ord', 'StemBr2ord', 'StemL1ord', 'LeafSize',
              'PodD', 'PodS', 'AscoRes', 'Germ']


path_data = '../data/cv_model/norm/'
path_cv = '../data/cv_model/'
path_mod = '../data/cv_model/models/'
path_mod_res = '../data/cv_model/mod_grow/'
n_cv = 20

for idata in [1]:
    file_data = 'data_train_' + str(idata) + '.txt'
    file_cov = 'cov_mx_all_' + str(idata) + '.txt'
    mx_cov = read_csv(path_cv + file_cov, sep='\t', header=0, index_col=0)

    for thresh in [5]:


        file_mod = 'mod_' + str(thresh) + '_' + str(idata) + '.txt'

        path_res = path_cv + 'mod_grow/res_' + str(idata) + '_' + str(thresh) +  '/'
        try:
            os.mkdir(path_res)
        except:
            print('Directory is already created')


        mod = grow_pipeline(path_mod, file_mod, path_data, file_data, path_res,
                            idata=idata, mx_cov=mx_cov)
        show(mod)

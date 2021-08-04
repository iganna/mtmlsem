"""
Run MCMC
"""

__author__ = "Anna Igolkina"
__license__ = "MIT"
__maintainer__ = "Anna Igolkina"
__email__ = "igolkinaanna11@gmail.com"

import numpy as np
import random

from pandas import read_csv
from multiprocess import Pool
from itertools import product

from utils import create_path, sort_second
from func_bayes import ModelBay, SEMmx




def mcmc_individual(idata, nseed):
    print(idata, nseed)

    file_mod = ('mod_spart_' if beta_flag else 'mod_') + str(thresh) + '_' + str(idata) + '.txt'
    file_data = 'data_train_' + str(idata) + '.txt'
    file_prior = 'prior_' + str(thresh) + '_' + str(idata) + '.txt'
    file_mcmc =  'mcmc_' + str(thresh) + '_' + str(idata) + '.txt'

    # -------------------------------------------------------------------

    with open(path_mod_specif + file_mod, 'r') as f:
        mod = f.read()

    param_prior = read_csv(path_priors + file_prior, sep='\t', header=None)
    param_prior = param_prior.iloc[:, :3]
    data = read_csv(path_cv_data + file_data, sep='\t', header=0, index_col=0)


    mod_bay = ModelBay(mod, data, param_prior, var_ordinal, kappa_flag=kappa_flag)
    self = mod_bay

    
    random.seed(nseed)
    self.optimise()


    # -------------------------------------------------------------------
    mcmc = mod_bay.mcmc
    np.savetxt(path_mcmc_seed + file_mcmc, mcmc)
    
def mcmc(idata):   

    rseeds = [125, 137, 159, 257, 31] 
    for iseed in range(len(rseeds)):
        path_mcmc_seed = path_mcmc + '{}/'.format(iseed)
        create_path(path_mcmc_seed)
        print(path_mcmc_seed)
        
        mcmc_individual(idata, rseeds[iseed])
        


var_ordinal = ['FloCol', 'StemCol', 'FlowStemCol', 'SeedCol', 'PodShape', 'SeedShape',
               'BushShape', 'StemBr', 'StemBr1ord', 'StemBr2ord', 'StemL1ord', 'LeafSize',
              'PodD', 'PodS', 'AscoRes', 'Germ']


n_cv = 20  # Number of cross-validation sets
thresh = 5 # Number of latent factors

kappa_flag = True  # True - if you are using extended model; False - if base model
beta_flag = True  # True - if connected model; False - if zero model

path_cv_data = '../data/cv_data/'  # path with training and test sets
path_cv_model = '../data/cv_model/'  # path with modes and para,eters
path_mod_specif = path_cv_model + ('mod_grow/' if beta_flag else 'models/')  # path with model specification
path_priors = path_cv_model + 'priors_' + \
                ('full_' if beta_flag else 'zero_') + 'cut' + \
                ('_phen/' if kappa_flag else '/')  # prior values of parameters


path_mcmc = path_cv_model + 'mcmc_final_' + \
                ('full_' if beta_flag else 'zero_') + 'cut' + \
                ('_phen_' if kappa_flag else '_') # mcmc chains


    

n_thr = 10
with Pool(n_thr) as workers:
    pmap = workers.map
    pmap(mcmc, range(n_cv))


# -----------------------------------------------------------
# -----------------------------------------------------------


iseed = 0
path_mcmc_seed = path_mcmc + '{}/'.format(iseed)

# path_mcmc_seed = '../data/cv_model/mcmc_zero_cut_phen/'

data_res = []
data_test = []
for idata in range(n_cv):
    print(idata)


    file_mod = ('mod_spart_' if beta_flag else 'mod_') + str(thresh) + '_' + str(idata) + '.txt'
    file_data = 'data_train_' + str(idata) + '.txt'
    file_prior = 'prior_' + str(thresh) + '_' + str(idata) + '.txt'
    file_mcmc =  'mcmc_' + str(thresh) + '_' + str(idata) + '.txt'



    # -------------------------------------------------------------------

    with open(path_mod_specif + file_mod, 'r') as f:
        mod = f.read()

    param_prior = read_csv(path_priors + file_prior, sep='\t', header=None)
    param_prior = param_prior.iloc[:, :3]
    data = read_csv(path_cv_data + file_data, sep='\t', header=0, index_col=0)
    # -------------------------------------------------------------------

    mod_bay = ModelBay(mod, data, param_prior, var_ordinal, kappa_flag=kappa_flag)
    self = mod_bay

    # -------------------------------------------------------------------

    mcmc = read_csv(path_mcmc_seed + file_mcmc, sep=' ', header=None)
    x = np.mean(mcmc[100:1000], axis=0)
    self.param_val = x
    self.fill_mx()

    t_data = data.copy()
    tmp = self.predict_new(t_data)


    file_t_data = 'data_test_' + str(idata) + '.txt'
    t_data = read_csv(path_cv_data + file_t_data, sep='\t', header=0, index_col=0)
    y1, y2 = self.predict_new(t_data)

    y1 = y1 * 2
    data_res += [y1]
    data_test += [y2]


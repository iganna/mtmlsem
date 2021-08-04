
import os
import numpy as np
import pandas as pd

from mtml_model import mtmlModel
from dataset import Data, CVset
from pandas import read_csv, DataFrame
from optimisation import *
import semopy
import random
import semba
from utils import *

from semopy import Model as semopyModel
from lat_struct import *
from optimisation import *
from utils import *
from unit_tests import test0, test1, test2


# model_desc, data, model = test0()
#
# model_desc, data, model = test1()
#
# model_desc, data, model = test2()



# Generate dataset

np.random.seed(125)
random.seed(239)
n = 100

model_desc = semopy.model_generation.generate_desc(n_lat=3, n_endo=0, n_exo=0, n_inds=3, n_cycles=0)
show(model_desc)
params, aux = semopy.model_generation.generate_parameters(model_desc)
data_gen = semopy.model_generation.generate_data(aux, n)
data_gen.index = [f's{i}' for i in range(n)]

# Generate SNPs
n_snps = 1000
data_snps = np.random.binomial(2, 0.5, size=(n,n_snps))
for i in range(20):
    tmp = data_snps[:,i] + 0
    j = np.random.binomial(data_gen.shape[1], 0.5, size=1)[0]

    order = np.argsort(data_gen.iloc[:,j].to_numpy())
    ranks = np.argsort(order)
    data_snps[:, i] = tmp[ranks]
data_snps = DataFrame(data_snps, index=[f's{i}' for i in range(n)],
                      columns=[f'snp_{i}' for i in range(n_snps)])


data = Data(d_phens=data_gen, d_snps=data_snps)

model = mtmlModel(model_desc=model_desc,
                  data=data)

model = mtmlModel(data=data)

# opt = model.opt_bayes()
# print(model.unnormalize())

# semopy.efa.explore_cfa_model(data.d_phens)

model.get_lat_struct(remain_models=True)
model.show_mod()

mod = model.mods['mod0']
variable = 'F0'

defenerate_flag = True

model.add_snps()







x = np.concatenate((opt.data['_observed'], opt.data['group1']), axis=1)
np.corrcoef(x.transpose())
opt.v



model = mtmlModel(model_desc=model_desc,
                  data=data)

opt1 = model.opt_bayes()

print(model.unnormalize_params())

#
#
#
#
print(params)

sem_old = Model('x1 ~ g1 + g2 + group1')
sem_old.fit(data_gen)
print(sem_old.inspect())

sem_old = Model('x1 ~ g1 + g2')
sem_old.fit(data_gen)
print(sem_old.inspect())

semba_model = semba.Model(model_desc)
semba_model.fit(data_gen, num_samples=1000, progress_bar=False)
print('semba')
insp = semba_model.inspect()
insp = insp.loc[insp['op'] == '~', ['lval', 'rval', 'Estimate']]
print(insp)

#
#
#
show(model_desc)
semopy_model = ModelEffects(model_desc, effects=['group1'])
semopy_model.fit(data_gen)
print(semopy_model.inspect())



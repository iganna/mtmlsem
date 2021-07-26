
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
from func_util import *

from semopy import Model as semopyModel
from lat_struct import *
from optimisation import *
from func_util import *
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

model_desc = semopy.model_generation.generate_desc(n_lat=0, n_endo=1, n_exo=2, n_inds=3, n_cycles=0)
# show(model_desc)
params, aux = semopy.model_generation.generate_parameters(model_desc)
data_gen = semopy.model_generation.generate_data(aux, n)
data_gen.index = [f's{i}' for i in range(n)]

# generate random effects
group1 = pd.DataFrame(data={'group1': np.random.binomial(2, 0.5, size=n)})
group1.index = [f's{i}' for i in range(n)]

model_desc = """
x1 ~ g1 + g2 + group1
"""

# model_desc = """
# x1 ~ g2 + group1
# """

show(model_desc)

tmp = 0 + group1['group1']
tmp[tmp == 1] = 3
data_gen['x1'] = data_gen['x1'] + 2 * tmp
data_gen = concat([data_gen, group1], axis=1)


data = Data(d_phens=data_gen,
            s_reffs=['group1'],
            show_warning=False)


model = mtmlModel(model_desc=model_desc,
                  data=data,
                  random_effects=['group1'])

np.random.seed(2346)
opt = model.opt_bayes()

print(model.unnormalize())

x = np.concatenate((opt.data['_observed'],opt.data['group1']), axis=1)
np.corrcoef(x.transpose())
opt.v



model = mtmlModel(model_desc=model_desc,
                  data=data)

opt1 = model.opt_bayes()

print(model.unnormalize())

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
semba_model.fit(data_gen, num_samples=1000)
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



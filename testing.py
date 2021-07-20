
import os
import numpy as np
import pandas as pd

from mtml_model import mtmlModel
from dataset import Data, CVset
from pandas import read_csv
from optimisation import *
import semopy
import semba
from func_util import *

from semopy import Model as semopyModel
from lat_struct import *
from optimisation import *
from func_util import *


# Generate dataset
np.random.seed(125)
n = 100

model_desc = semopy.model_generation.generate_desc(n_lat=1, n_endo=1, n_exo=4, n_inds=3, n_cycles=0)
show(model_desc)
params, aux = semopy.model_generation.generate_parameters(model_desc)
data_gen = semopy.model_generation.generate_data(aux, n)
data_gen.index = [f's{i}' for i in range(n)]

# model_desc = """
# eta1 =~ y1 + y2 + y3
# """

snp = np.random.binomial(2, 0.5, size=n)
snps = pd.DataFrame(data={'snp': snp})
snps.index = [f's{i}' for i in range(n)]


data = Data(d_phens=data_gen, d_snps=snps)



model = mtmlModel(model_desc=model_desc,
                  data=data)
# self = model

model.opt_bayes()

print(model.unnormalize())


# -----------------------------------
#
# sem_old = Model(model_desc)
# sem_old.fit(data=data.d_phens)
# sem_old.inspect()
#
#
sem_old = Model(model_desc)
sem_old.fit(data_gen)
print('semopy')
print(sem_old.inspect())
#
#
# sem_old = Model(model_desc)
# sem_old.fit(data=data.d_phens)
# # sem_old.fit(data=data_gen)
# sem_old.inspect()

#
#
semba_model = semba.Model(model_desc)
semba_model.fit(data_gen, num_samples=1000)
print('semba')
print(semba_model.inspect())
#


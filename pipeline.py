
import os
from mtml_model import mtmlModel
from dataset import Data, CVset
from pandas import read_csv
from optimisation import *

from func_util import *

path_data = 'data_cicer/'
file_phens = path_data + 'data_phens.txt'

file_snps = path_data + 'snp_2579_renamed.txt'
# file_model = path_data + 'model_05.txt'
# file_model = path_data + 'model_connected.txt'

file_model = path_data + 'model_test3.txt'

file_phen_types = path_data + 'phen_types.txt'


data = Data(file_phens=file_phens,
           file_phen_types=file_phen_types,
           file_snps=file_snps)


model = mtmlModel(model_file=file_model,
                  data=data)


# model.get_lat_struct(cv=True)

opt = OptBayes(sem=model.sems, data=data)

self = opt

insp = sem.inspect()

for i in range(insp.shape[0]):
    print(f'{insp.lval[i]} {insp.op[i]} {insp.rval[i]}')


# mod = model.sem_model.description
#
# show(mod)
#
# variable = 'FC1'
#
# d_phens = data.d_phens
#

# Cross-validates datasets
n_cv = 10
cv_data = CVset(dataset=data, n_cv=n_cv)



d1 = cv_data.train[1]


mod_new = model.add_snps()

#
# descr = """ FC1 ~ FC2
# FC4 ~ FC3
# FC1 ~ FC4
# FC5 ~ FC1 + FC2
# """


from semopy.examples import example_rf
from semopy import ModelEffects

desc = example_rf.get_model()
data, k = example_rf.get_data()

model = ModelEffects(desc)
model.fit(data, group='group', k=k)
print(model.inspect())
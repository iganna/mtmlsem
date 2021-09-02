import os
import numpy as np
import pandas as pd

from .mtml_model import mtmlModel
from .dataset import Data, CVset
from pandas import read_csv
from .optimisation import *
import semopy
import semba
import random
from .utils import *

from semopy import Model as semopyModel
from .lat_struct import *
from .optimisation import *
from .utils import *




def test0():

    # Generate dataset
    np.random.seed(125)
    random.seed(239)
    n = 100


    model_desc = semopy.model_generation.generate_desc(n_lat=0, n_endo=1, n_exo=2, n_inds=3, n_cycles=0)
    params, aux = semopy.model_generation.generate_parameters(model_desc)
    data_gen = semopy.model_generation.generate_data(aux, n)
    data_gen.index = [f's{i}' for i in range(n)]

    # generate random effects
    group1 = pd.DataFrame(data={'group1': np.random.binomial(1, 0.5, size=n)})
    group1.index = [f's{i}' for i in range(n)]

    model_desc = """
    x1 ~ g1 + g2 + group1
    """
    #
    # model_desc = """
    # x1 ~ g1 + g2
    # """

    data_gen['x1'] = data_gen['x1'] + 10 * group1['group1']
    data_gen = concat([data_gen, group1], axis=1)

    data = Data(d_phens=data_gen,
                show_warning=False)


    model = mtmlModel(model_desc=model_desc,
                      data=data)
    model.opt_bayes()

    print(model.unnormalize_params())

    # -----------------------------------
    # semopy
    sem_old = Model(model_desc)
    sem_old.fit(data_gen)
    print('semopy')
    insp = sem_old.inspect()
    insp = insp.loc[insp['op'] == '~', ['lval', 'rval', 'Estimate']]
    print(insp)

    # semba
    semba_model = semba.Model(model_desc)
    semba_model.fit(data_gen, num_samples=1000)
    print('semba')
    insp = semba_model.inspect()
    insp = insp.loc[insp['op'] == '~', ['lval', 'rval', 'Estimate']]
    print(insp)
    print('params')
    print(params)

    return model_desc, data, model



def test1():

    # Generate dataset
    np.random.seed(125)
    random.seed(239)
    n = 100

    model_desc = semopy.model_generation.generate_desc(n_lat=1, n_endo=0, n_exo=0, n_inds=3, n_cycles=0)
    # show(model_desc)
    params, aux = semopy.model_generation.generate_parameters(model_desc)
    data_gen = semopy.model_generation.generate_data(aux, n)
    data_gen.index = [f's{i}' for i in range(n)]

    model_desc = """
    eta1 =~ y1 + y2 + y3
    """

    data = Data(d_phens=data_gen)

    model = mtmlModel(model_desc=model_desc,
                      data=data)
    model.show_mod()
    # self = model

    model.opt_bayes()

    print(model.unnormalize_params())

    # -----------------------------------
    # semopy
    sem_old = Model(model_desc)
    sem_old.fit(data_gen)
    print('semopy')
    insp = sem_old.inspect()
    insp = insp.loc[insp['op'] == '~', ['lval', 'rval', 'Estimate']]
    print(insp)

    # semba
    semba_model = semba.Model(model_desc)
    semba_model.fit(data_gen, num_samples=1000)
    print('semba')
    insp = semba_model.inspect()
    insp = insp.loc[insp['op'] == '~', ['lval', 'rval', 'Estimate']]
    print(insp)
    print('params')
    print(params)

    return model_desc, data, model




def test2():

    # Generate dataset
    np.random.seed(125)
    random.seed(239)
    n = 100

    model_desc = semopy.model_generation.generate_desc(n_lat=1, n_endo=1, n_exo=4, n_inds=3, n_cycles=0)
    # show(model_desc)
    params, aux = semopy.model_generation.generate_parameters(model_desc)
    data_gen = semopy.model_generation.generate_data(aux, n)
    data_gen.index = [f's{i}' for i in range(n)]

    # model_desc = """
    # x1 ~ g1 + g2
    # """

    # generate random effects
    group1 = pd.DataFrame(data={'group1': np.random.binomial(1, 0.5, size=n)})
    group1.index = [f's{i}' for i in range(n)]

    # model_desc = """
    # x1 ~ g1 + g2 + group1
    # """
    data_gen['x1'] = data_gen['x1'] + 10 * group1['group1']

    data_gen = concat([data_gen, group1], axis=1)

    snp = np.random.binomial(2, 0.5, size=n)
    snps = pd.DataFrame(data={'snp': snp})
    snps.index = [f's{i}' for i in range(n)]

    data = Data(d_phens=data_gen, d_snps=snps)

    model = mtmlModel(model_desc=model_desc,
                      data=data)
    model.show_mod()
    # self = model

    model.opt_bayes()

    print(model.unnormalize_params())

    # -----------------------------------
    # semopy
    sem_old = Model(model_desc)
    sem_old.fit(data_gen)
    print('semopy')
    insp = sem_old.inspect()
    insp = insp.loc[insp['op'] == '~', ['lval', 'rval', 'Estimate']]
    print(insp)

    # semba
    semba_model = semba.Model(model_desc)
    semba_model.fit(data_gen, num_samples=1000)
    print('semba')
    insp = semba_model.inspect()
    insp = insp.loc[insp['op'] == '~', ['lval', 'rval', 'Estimate']]
    print(insp)

    print('params')
    print(params)


    return model_desc, data, model



def test3():
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

    print(model.unnormalize_params())

    # x = np.concatenate((opt.data['_observed'], opt.data['group1']), axis=1)
    # np.corrcoef(x.transpose())
    # opt.v
    #
    # model = mtmlModel(model_desc=model_desc,
    #                   data=data)
    #
    # opt1 = model.opt_bayes()
    #
    # print(model.unnormalize())

    #
    #
    #
    #
    # print(params)
    print('semopy with reff as variable')
    sem_old = Model('x1 ~ g1 + g2 + group1')
    sem_old.fit(data_gen)
    print(sem_old.inspect())

    print('semopy without reff variable')
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

    print('params')
    print(params)
    return  model_desc, data, model
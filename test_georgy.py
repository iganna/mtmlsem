
import os
import numpy as np
import pandas as pd

import semba
import semopy

np.random.seed(239)
n = 100
x = np.random.normal(size=n)
y = np.random.normal(size=n)
k = np.random.normal(size=n)
e1 = np.random.normal(size=n)/100
e2 = np.random.normal(size=n)/100
z = 5 * x + 2 * y + e1
t = 6 * x + 3 * y + 10 * k + e2
snp = np.random.binomial(2, 0.5, size=n)

phens = pd.DataFrame(data={'x': x, 'y': y, 'z': z, 'e1': e1,
                           't': t, 'e2': e2, 'k': k})
phens.index = [f's{i}' for i in range(n)]


model_desc = """
t ~ x + y + k"""

# from semopy.examples import political_democracy as ex

#
# desc, data = ex.get_model(), ex.get_data()

sem_old = semopy.Model(model_desc)
sem_old.fit(phens)
sem_old.inspect()



model = semba.Model(model_desc)
model.fit(phens)
model.inspect()


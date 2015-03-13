from __future__ import print_function
import numpy as np

with open('form_factors_2body.dat') as f:

    names = f.readline().split()[1:]
    data = [[float(x) for x in line.split()] for line in f]

    nvalues = len(data)

    dtype = zip(names, [np.float64]*nvalues)
    np_data = np.zeros(nvalues, dtype)

    data = np.asarray(data).T
    for n, name in enumerate(names):
        np_data[name] = data[n]
    np_data.dump('form_factors_2body.npy')

with open('form_factors_1body.dat') as f:

    names = f.readline().split()[1:]
    data = [[float(x) for x in line.split()] for line in f]

    nvalues = len(data)

    dtype = zip(names, [np.float64]*nvalues)
    np_data = np.zeros(nvalues, dtype)

    data = np.asarray(data).T
    for n, name in enumerate(names):
        np_data[name] = data[n]
    np_data.dump('form_factors_1body.npy')

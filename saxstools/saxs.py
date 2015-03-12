import os.path
import numpy as np
from scipy.interpolate import interp1d

FORM_FACTORS_2BODY = np.load(os.path.join(os.path.dirname(__file__), 'data', 'form_factors_2body.npy'))

def form_factor(q, entity):
    
    # form factor function
    fff = interp1d(FORM_FACTORS_2BODY['q'], FORM_FACTORS_2BODY[entity], 
            copy=False, kind='cubic', assume_sorted=True)

    return fff(q)


def scattering_curve(q, entities, ind_entities, out=None):
    pass

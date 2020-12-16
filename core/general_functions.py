import numpy as np
from constants import *

def to_db(lin_value):
    return 10*np.log10(lin_value)

def to_linear(db_value):
    return 10**(db_value/10)

def to_alpha_linear(alpha_db):
    return alpha_db/(20*np.log10(e))
import numpy as np

def to_db(lin_value):
    return 10*np.log10(lin_value)

def to_linear(db_value):
    return 10**(db_value/10)

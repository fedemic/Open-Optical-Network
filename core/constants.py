import numpy as np
from scipy.constants import c
from scipy.constants import h
from scipy.constants import pi

e = np.exp(1)

N_CHANNELS = 10
BER_T = 1e-3        # bit error rate
RS = 32e9           # symbol rate [baud/s]
DF = 50e9         # channel spacing [Hz]
BN = 12.5e9         # noise bandwidth [Hz]
F = 193.414e12      # C band center frequency [Hz]

AMP_GAIN = 16       # [dB]
AMP_NF = 3          # [dB]

ALPHA = 0.2         # [dB/km]
BETA_2 = 2.13e-26   # [(m*Hz^2)^-1]
GAMMA = 1.27e-3     # [(W*m)^-1]

N_CONNECTIONS = 2
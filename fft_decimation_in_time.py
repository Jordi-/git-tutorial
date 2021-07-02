import numpy as np

# Basic demo of
# 1. Given a secuence, compute and apply a Hann Window
# 2. Compute DFT using the decimation in time algorithm => FFT

x = np.array([1, 1, 2, 0, 0, 2, 1, 1])
L = len(x)
N = L - 1

# Hanning window
w = 0.5 * (1 - np.cos(2*np.pi*np.arange(L)/N))

# Windowed signal
xw = x * w


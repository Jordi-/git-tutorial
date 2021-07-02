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

# Wkn
Wkn = np.exp(-1j * 2*np.pi * np.arange(4)/L)

# Step 1 combine first with second half of the sequence
X1 = np.zeros(L)
X1[0:4] = xw[0:4] + xw[4:]
X1[4:8] = xw[0:4] - xw[4:]

print('Pass 1a', X1)
# Multiply second half by Wkn
X1 = X1 * np.hstack((np.ones_like(Wkn), Wkn))
print('Pass 1b', X1)

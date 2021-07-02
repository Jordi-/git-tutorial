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
# Patch: Wkn[2] is super ugly!
Wkn[2] = -1j

# Step 1: combine first with second half of the sequence
X1 = np.zeros(L)
X1[0:4] = xw[0:4] + xw[4:]
X1[4:8] = xw[0:4] - xw[4:]

print('Pass 1a', X1)
# Multiply second half by Wkn
X1 = X1 * np.hstack((np.ones_like(Wkn), Wkn))
print('Pass 1b', X1)

# Step 2: 4 elements combos
X2 = np.zeros(L)
X2 = X2 + 1j * 0  # Trick to convert X2 to complex
X2[0:2] = X1[0:2] + X1[2:4]
X2[2:4] = X1[0:2] - X1[2:4]
X2[4:6] = X1[4:6] + X1[6:8]
X2[6:8] = X1[4:6] - X1[6:8]

print('Pass 2a', X2)
# Multiply by Wkn factors
X2 = X2 * np.array([1, 1, Wkn[0], Wkn[2], 1, 1, Wkn[0], Wkn[2]])
print('Pass 2b', X2)

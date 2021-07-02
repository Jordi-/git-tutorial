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

# Step 3: 2 elements combos
X3 = np.zeros_like(X2)  # This directly creates a complex array
X3[0] = X2[0] + X2[1]
X3[1] = X2[0] - X2[1]
X3[2] = X2[2] + X2[3]
X3[3] = X2[2] - X2[3]
X3[4] = X2[4] + X2[5]
X3[5] = X2[4] - X2[5]
X3[6] = X2[6] + X2[7]
X3[7] = X2[6] - X2[7]

# Bit reverse
# binary_repr(...)[::-1] returns the binary number reversed
# int(..., base=2) converts back to decimal
br = [int(np.binary_repr(x, width=3)[::-1], base=2) for x in range(8)]
X3 = X3[br]

# Check result
print('FFT      ', X3)
print('FFT numpy', np.fft.fft(xw))

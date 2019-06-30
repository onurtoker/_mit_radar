"""Time resampling test using recorded data
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
import scipy.io.wavfile

fs = 44.1e3
NS = round(10 * fs / 1000)
fLen = 2**16
Ts = 1 / fs

rate, dataX = scipy.io.wavfile.read('data/delayLong_10ms.wav')
dataX = dataX.astype(np.float)

chC = dataX[:,0]
chT = dataX[:,1]
dataC = chC.reshape((-1,))
dataT = chT.reshape((-1,))

tvals = np.arange(NS) * Ts
fvals = fftfreq(fLen, 1/fs)

# (re)find the beginning of the sweep
ni = NS // 2    # skip some samples
st = dataT[ni: ni + NS]
ni = ni + np.argmax(np.diff(st))

y_vals = dataC[ni : ni + NS]
#st = dataT[ni : ni + NS]
#ni = ni + NS

# SW resampling
pvals = np.unwrap(np.angle(hilbert(y_vals)))
puni_vals = np.linspace(np.min(pvals), np.max(pvals), len(pvals))
tr_vals = np.interp(puni_vals, pvals, tvals)  # A higher resolution VCO info will be better
yS_vals = np.interp(tr_vals, tvals, y_vals)

# FFT comparison
Y = np.abs(fft(y_vals, fLen))
YS = np.abs(fft(yS_vals, fLen))

Y = Y / np.max(Y)
YS = YS / np.max(YS)

Y = fftshift(Y)
YS = fftshift(YS)
fvals = fftshift(fvals)

Y=Y[fLen//2:]
YS=YS[fLen//2:]
fvals=fvals[fLen//2:]

# Plots
plt.subplot(1,2,1)
plt.plot(tvals, y_vals, tvals, yS_vals)
plt.title('IF signals')
plt.xlabel('time (s)')

plt.subplot(1,2,2)
plt.plot(fvals, Y, fvals, YS)
plt.title('IF spectrums')
plt.xlabel('freq (Hz)')

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import pinv, norm
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
import scipy.io.wavfile
# import threading
# from six.moves import queue

# system parameters
fs = 44.1e3
td = 10e-3
NS = round(td * fs)
fLen = NS #2**16
BW = 250e6
c = 3e8

# read data from an audio file
rate, dataX = scipy.io.wavfile.read('data/dL_10ms.wav')
dataX = dataX.astype(np.float)

# isolate channels
chC = dataX[:,0]
chT = dataX[:,1]
dataC = chC.reshape((-1,))
dataT = chT.reshape((-1,))

# generate animated figure
fig = plt.figure()

tv = np.linspace(-500 * td, 0, 500)
fv = fftfreq(fLen, 1/fs)[:fLen//2]
dv = fv * (c * td / 2 / BW)
Z = np.zeros((len(dv),len(tv)))
im = plt.imshow(Z, animated=True, vmin = 0, vmax = +1, origin='lower')

# skip some samples
ni = NS // 2

# update figure function
def updatefig(*args):
    global Z, ni

    #print(ni)
    # re-find the beginning of the sweep (Backup 5 samples)
    ni = ni - 5
    st = dataT[ni: ni + NS]
    ni = ni + np.argmax(np.diff(st))
    # skip the first ni samples to reach to the beginning of the sweep
    sc = dataC[ni: ni + NS]
    ni = ni + NS

    Y = np.abs(fft(sc, fLen))
    Y = Y / np.max(Y)
    Y = Y[:fLen//2]

    Z = np.hstack((Z[:, 1:], Y.reshape((-1,1))))
    im.set_array(Z)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()





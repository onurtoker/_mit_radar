import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
import scipy.io.wavfile

fs = 44100
NS = 441
fLen = 2**14
gLen = 250

rate, dataC = scipy.io.wavfile.read('data/delayLine.wav')

chC = dataC[:,0]
dataC = chC.reshape((-1,))

f = fftfreq(fLen, 1/fs)

fig, pm  = plt.subplots(2, 1)
sc_fig = pm[0].plot(np.zeros(NS))[0]
pc_fig = pm[1].plot(np.zeros(NS - 1))[0]

pm[0].set_ylim(-2500,2500)
pm[0].set_ylabel('IF signal')
pm[1].set_ylim(20,50)
pm[1].set_ylabel('diff(IF phase)')
pm[1].set_xlabel('sample index')
plt.draw()
plt.pause(0.005)

ni = 535    # skip the first ni samples to find the beginning of cycles
while True:

    sc = dataC[ni : ni + NS]
    ni = ni + NS
    sc_fig.set_ydata(sc)

    pc = np.unwrap(np.angle(hilbert(sc)))
    pc_fig.set_ydata(np.diff(pc) * 180 / np.pi)

    # pm[1].relim()
    # pm[1].autoscale_view()

    plt.draw()
    plt.pause(0.001)

while True:
    pass




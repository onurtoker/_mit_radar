import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
import scipy.io.wavfile

fs = 44100
NS = fs // 10
fLen = NS #2**14
gLen = 250

rate, dataC = scipy.io.wavfile.read('/home/jetson/PythonProjects/mit_radar-master/delayLong_100ms.wav')

chC = dataC[:,0] 
dataC = chC.reshape((-1,))

f = fftfreq(fLen, 1/fs)

fig, pm  = plt.subplots(2, 1)
sc_fig = pm[0].plot(np.zeros(NS))[0]
pc_fig = pm[1].plot(np.zeros(fLen))[0]

pm[0].set_ylim(-3200,3200)
pm[0].set_ylabel('IF signal')
pm[1].set_ylim(0,360)
pm[1].set_ylabel('diff(IF phase)')
pm[1].set_xlabel('sample index')
plt.draw()
plt.pause(0.005)

ni = 126   # skip the first ni samples to find the beginning of cycles

sc = dataC[ni : ni + NS]
ni = ni + NS
sc_fig.set_ydata(sc)

pc = np.unwrap(np.angle(hilbert(np.pad(sc, (0,fLen-NS), 'constant'))))
pc_fig.set_ydata(pc) #np.diff(pc) * 180 / np.pi)

plt.draw()
plt.show()

while True:
    pass


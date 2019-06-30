import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
import scipy.io.wavfile

fs = 44.1e3
NS = round(10 * fs / 1000)
fLen = 2**16

rate, dataX = scipy.io.wavfile.read('data/delayLong_10ms.wav')
dataX = dataX.astype(np.float)

chC = dataX[:,0]
chT = dataX[:,1]
dataC = chC.reshape((-1,))
dataT = chT.reshape((-1,))

f = fftfreq(fLen, 1/fs)

fig, pm  = plt.subplots(2, 1)
sc_fig = pm[0].plot(np.zeros(NS))[0]
pc_fig = pm[1].plot(np.zeros(NS))[0]

#pm[0].set_ylim(-32000,32000)
pm[0].set_ylabel('IF signal')
#pm[1].set_ylim(-10,10)
pm[1].set_ylabel('(IF phase)')
pm[1].set_xlabel('sample index')
pLabel = pm[1].text(0, 10, '')

plt.draw()
plt.pause(0.005)

# skip some samples
ni = NS // 2

scaled = False
try:
    while True:
        # re-find the beginning of the sweep (Backup 5 samples)
        ni = ni - 5
        st = dataT[ni: ni + NS]
        ni = ni + np.argmax(np.diff(st))

        # skip the first ni samples to find the beginning of cycles
        sc = dataC[ni : ni + NS]
        st = dataT[ni : ni + NS]
        ni = ni + NS
        sc_fig.set_ydata(sc)

        pc = np.unwrap(np.angle(hilbert(sc)))
        pc = pc - pc[0]
        pc_fig.set_ydata(pc)
        #pc_fig.set_ydata(np.diff(pc) * 180 / np.pi)

        if (scaled == False):
            pm[0].relim()
            pm[0].autoscale_view()
            pm[1].relim()
            pm[1].autoscale_view()
            scaled = True

        pLabel.set_text("{0:d}".format(ni // NS))

        plt.draw()
        plt.pause(0.001)
except:
    # ni = ni - 10 * NS
    # st = dataT[ni: ni + NS]
    # ni = ni + np.argmax(np.diff(st))
    # sc = dataC[ni: ni + NS]
    # sc_fig.set_ydata(sc)
    #
    # pc = np.unwrap(np.angle(hilbert(sc)))
    # pc = pc - pc[0]
    # pc_fig.set_ydata(pc)
    #
    # pLabel.set_text("{0:d}".format(ni // NS))

    plt.draw()
    plt.show()




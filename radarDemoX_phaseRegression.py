import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
import scipy.io.wavfile

fs = 44100
NS = 441*4
fLen = 2**14
gLen = 250

rate, dataC = scipy.io.wavfile.read('C:\\Users\\otoker\\Desktop\\Projects\\PyProjects\\radar\\delayLine.wav')
rate, dataX = scipy.io.wavfile.read('C:\\Users\\otoker\\Desktop\\Projects\\PyProjects\\radar\\near.wav')
#rate, dataX = scipy.io.wavfile.read('C:\\Users\\otoker\\Desktop\\Projects\\PyProjects\\radar\\far.wav')

chX = dataX[:,0]
dataX = chX.reshape((-1,))
chC = dataC[:,0]
dataC = chC.reshape((-1,))

f = fftfreq(fLen, 1/fs)

fig, pm  = plt.subplots(3, 2)
sb_fig = pm[0,0].plot(np.zeros(NS))[0]
Sb_fig = pm[1,0].plot(f[:fLen // 2], np.zeros(fLen // 2))[0]
est1_fig = pm[2,0].plot(np.arange(-gLen,0), np.zeros(gLen))[0]

sc_fig = pm[0,1].plot(np.zeros(NS))[0]
Sc_fig = pm[1,1].plot(f[:fLen // 2], np.zeros(fLen // 2))[0]
est2_fig = pm[2,1].plot(np.arange(-gLen,0), np.zeros(gLen))[0]

pm[0,0].set_ylim(-2500,2500)
pm[0,0].set_ylabel('IF signal')
pm[1,0].set_ylim(0, 1.2)
pm[1,0].set_xlim(0,10000)
pm[1,0].set_ylabel('IF spectrum')
pm[2,0].set_ylabel('PeakLog')

pm[0,1].set_ylim(-2500,2500)
pm[0,1].set_ylabel('IF signal')
pm[1,1].set_ylim(0, 1.2)
pm[1,1].set_xlim(0,10000)
pm[1,1].set_ylabel('IF spectrum')
pm[2,1].set_ylabel('PeakLog')

plt.draw()
plt.pause(0.005)

ni = 535    # skip the first ni samples to find the beginning of cycles
while True:

    # TARGET BEAT SIGNAL DSP
    sb = dataX[ni : ni + NS]
    ni = ni + NS

    sb_fig.set_ydata(sb)

    Sb = fft(sb, fLen)
    Sb = np.abs(Sb[:fLen//2])
    Sb_fig.set_ydata(Sb/np.max(Sb))

    f_pk = f[np.argmax(Sb)]
    est1_fig.set_ydata(np.append(est1_fig.get_ydata()[-gLen + 1:], f_pk))
    pm[2,0].relim()
    pm[2,0].autoscale_view()

    # CALIBRATION BEAT SIGNAL DSP
    sc = dataC[ni : ni + NS]
    #ni = ni + NS

    sc_fig.set_ydata(sc)

    Sc = fft(sc, fLen)
    Sc = np.abs(Sc[:fLen//2])
    Sc_fig.set_ydata(Sc/np.max(Sc))

    f_pk = f[np.argmax(Sc)]

    # BEGIN PHASE REGRESSION
    pX = np.unwrap(np.angle(hilbert(sb)))
    pC = np.unwrap(np.angle(hilbert(sc)))
    A = np.vstack([pC, np.ones(len(pC))]).T
    m, n = np.linalg.lstsq(A, pX, rcond=None)[0]
    f_pk = m * f_pk
    # END PHASE REGRESSION

    est2_fig.set_ydata(np.append(est2_fig.get_ydata()[-gLen + 1:], f_pk))
    pm[2,1].relim()
    pm[2,1].autoscale_view()

    plt.draw()
    plt.pause(0.001)

while True:
    pass




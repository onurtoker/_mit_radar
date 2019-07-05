import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.io.wavfile

fs = 44100
NS = 441
N1 = 1
N2 = 1
NC = N1 * N2
fLen = 2**14
gLen = 250

rate, dataX = scipy.io.wavfile.read('data/delayLine.wav')
#rate, dataX = scipy.io.wavfile.read('data/near.wav')
#rate, dataX = scipy.io.wavfile.read('data/far.wav')
dataX = dataX.astype(np.float)

ch1 = dataX[:,0]
dataX =  ch1.reshape((ch1.shape[0],))

f = scipy.fftpack.fftfreq(fLen, 1 / fs)
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
aSig = ax0.plot(np.zeros(NS))[0]
pSpectrum = ax1.plot(f[:fLen // 2], np.zeros(fLen // 2))[0]
pDistLog = ax2.plot(np.arange(-gLen,0), np.zeros(gLen))[0]
ax0.set_ylim(-2500,2500)
ax0.set_ylabel('IF signal')
ax1.set_ylim(0, 1.2)
ax1.set_xlim(0,10000)
ax1.set_ylabel('IF spectrum')
#pkLabel = ax2.text(0, 0, '')
#ax2.set_ylim(4990,5000)
#ax2.set_ylim(5250,5350)
ax2.set_ylabel('PeakLog')
plt.draw()
plt.pause(0.005)

ni = 535
while True:

    data = dataX[ni:ni + NC * NS]
    ni = ni + NC * NS

    #data = data.astype(float)
    mval = data.reshape(NC, NS)
    data = np.mean(mval, axis=0)
    aSig.set_ydata(data)

    #data = (data - scipy.mean(data)) * scipy.kaiser(len(data), 14)
    data = np.concatenate((data, np.zeros(fLen - NS)))
    S = scipy.fftpack.fft(data)
    S = np.abs(S[:fLen//2])
    pSpectrum.set_ydata(S/np.max(S))
    #pSpectrum.set_ydata(20*scipy.log10(S/np.max(S)))

    f_pk = f[np.argmax(S)]
    #pkLabel.set_text("{0:.2f}Hz".format(f_pk))
    pDistLog.set_ydata(np.append(pDistLog.get_ydata()[-gLen + 1:], f_pk))
    ax2.relim()
    ax2.autoscale_view()

    plt.draw()
    plt.pause(0.001)

while True:
    pass




import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

fs = 44100
NS = 441
N1 = 10
N2 = 10
NC = N1 * N2
fLen = 2**14
gLen = 250
NFB = 4096 * 100
#NFB = 2 * NS * NC

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs,
                input=True, frames_per_buffer=NFB, start=True)

f = scipy.fftpack.fftfreq(fLen, 1 / fs)
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
aSig = ax0.plot(np.zeros(NS))[0]
pSpectrum = ax1.plot(f[:fLen // 2], np.zeros(fLen // 2))[0]
pDistLog = ax2.plot(np.arange(-gLen,0), np.zeros(gLen))[0]
ax0.set_ylim(-800,800)
ax1.set_ylim(0, 1.2)
ax1.set_xlim(0,2000)
ax1.set_title('IF spectrum')
ax1.set_xlabel('freq (Hz)')
pkLabel = ax2.text(20, 600, '')
ax2.set_ylim(600,1200)
ax2.set_title('Peak location history')
plt.draw()
plt.pause(0.005)

while True:

    #stream.start_stream()
    data = np.frombuffer(stream.read(NC * NS, exception_on_overflow = False), dtype=np.int16)
    #stream.stop_stream()
    data = data.astype(float)
    mval = data.reshape(NC, NS)
    adata = np.mean(mval, axis=0)
    aSig.set_ydata(adata)

    #data = (data - scipy.mean(data)) * scipy.kaiser(len(data), 14)
    adata = np.concatenate((adata[:NS], np.zeros(fLen - NS)))
    S = scipy.fftpack.fft(adata)
    S = np.abs(S[:fLen//2])
    pSpectrum.set_ydata(S/np.max(S))
    #pSpectrum.set_ydata(20*scipy.log10(S/np.max(S)))

    f_pk = f[np.argmax(S)]
    pkLabel.set_text("{0:.0f} Hz".format(f_pk))
    pDistLog.set_ydata(np.append(pDistLog.get_ydata()[-gLen + 1:], f_pk))

    plt.draw()
    plt.pause(0.005)

stream.stop_stream()
stream.close()
p.terminate()

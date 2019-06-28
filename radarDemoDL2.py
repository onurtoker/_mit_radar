# CHECK
# https://github.com/p5a0u9l/coffee-pi-radar/blob/master/alsa_serv.py

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from six.moves import queue

fs = 44100
NS = 441
N1 = 1
N2 = 1
NC = N1 * N2
fLen = 2**14
gLen = 250
NFB = 1024
#NFB = 256 * NS * NC
NU = 20

# Create a thread-safe buffer of audio data
buff = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status_flags):
    """Continuously collect data from the audio stream, into the buffer."""
    buff.put(in_data)
    return None, pyaudio.paContinue

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs,
                input=True, frames_per_buffer=NFB, start=True,
                stream_callback=audio_callback)

f = scipy.fftpack.fftfreq(fLen, 1 / fs)
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
aSig = ax0.plot(np.zeros(NS))[0]
pSpectrum = ax1.plot(f[:fLen // 2], np.zeros(fLen // 2))[0]
pDistLog = ax2.plot(np.arange(-gLen,0), np.zeros(gLen))[0]
ax0.set_ylim(-4e3,4e3)
ax1.set_ylim(0, 1.2)
ax1.set_xlim(0,10000)
ax1.set_title('IF spectrum')
ax1.set_xlabel('freq (Hz)')
pkLabel = ax2.text(0, 2400, '')
ax2.set_ylim(2400,3400)
ax2.set_title('Peak location history')
plt.draw()
plt.pause(0.005)

Su = []
nu = 0
while True:

    #stream.start_stream()
    ns = 0
    data = []
    buff.queue.clear()
    while (ns < NC * NS):
        chunk = np.frombuffer(buff.get(), dtype=np.int16)
        data.extend(chunk.tolist())
        ns += len(chunk)

    data = data[0:NC * NS]
    data = np.array(data)
    data = data.astype(float)

    #stream.stop_stream()
    mval = data.reshape(NC, NS)
    adata = np.mean(mval, axis=0)
    #aSig.set_ydata(adata)

    #data = (data - scipy.mean(data)) * scipy.kaiser(len(data), 14)
    adata = np.concatenate((adata[:NS], np.zeros(fLen - NS)))
    S = scipy.fftpack.fft(adata)
    S = np.abs(S[:fLen//2])
    Su.extend(S.tolist())
    nu += 1
    if (nu == NU):
        Su = np.array(Su)
        Su = Su.reshape(NU, fLen//2)
        Su = np.mean(Su, axis=0)

        pSpectrum.set_ydata(Su/np.max(Su))
        #pSpectrum.set_ydata(20*scipy.log10(S/np.max(S)))

        f_pk = f[np.argmax(Su)]
        pkLabel.set_text("{0:.2f}Hz".format(f_pk))
        pDistLog.set_ydata(np.append(pDistLog.get_ydata()[-gLen + 1:], f_pk))

        nu = 0
        Su = []
        plt.draw()
        plt.pause(0.005)

stream.stop_stream()
stream.close()
p.terminate()

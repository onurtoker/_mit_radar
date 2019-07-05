# CHECK
# https://github.com/p5a0u9l/coffee-pi-radar/blob/master/alsa_serv.py

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
from six.moves import queue

fs = 44100      # sampling frequency
NS = 2*441 #441 # number of samples
chirpLen = 441  # samples per chirp
NC = 1
fLen = 2**13
gLen = 250
NFB = 1024
NU = 1

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
aTrigger = ax0.plot(np.zeros(NS),'r-', alpha=0.5)[0]
pSpectrum = ax1.plot(f[:fLen // 2], np.zeros(fLen // 2))[0]
pDistLog = ax2.plot(np.arange(-gLen,0), np.zeros(gLen))[0]
ax0.set_ylim(-1.5,1.5)
ax1.set_ylim(0, 1.2)
ax1.set_xlim(0,10000)
ax1.set_title('IF spectrum')
ax1.set_xlabel('freq (Hz)')
pkLabel = ax2.text(0, 2500, '')
ax2.set_ylim(0000,6000)
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

    triggersig = scipy.diff(np.unwrap(np.angle(scipy.signal.hilbert(adata))))
    triggersig = triggersig-np.mean(triggersig)
    triggersig = triggersig/max(triggersig)
    peaks, _ = scipy.signal.find_peaks(triggersig, height=0.5)

    adata = (adata - scipy.mean(adata)) * scipy.kaiser(len(adata), 0)
    
    #aTrigger.set_ydata(np.pad(triggersig,(0,1),'constant'))
    aSig.set_ydata(adata/max(adata))
    if len(peaks) > 0:
        #chirpStart = np.zeros(chirpLen)
        adata = adata[peaks[0]:peaks[0]+chirpLen]
        adata = np.pad(adata,(0,chirpLen-len(adata)),'constant')
        aSig.set_ydata(np.pad(adata/max(adata),(0,NS-chirpLen),'constant'))
    
    #ax0.plot(peaks, triggersig[peaks], "x")

    S = scipy.fftpack.fft(adata,fLen)
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

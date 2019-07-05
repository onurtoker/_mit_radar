"""Dual channel radar demo"""

# Using thread-safe queue to buffer audio data
# Adapted from
# https://github.com/p5a0u9l/coffee-pi-radar/blob/master/alsa_serv.py

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import hilbert
##import scipy.fftpack
##import scipy.signal
import pyaudio
from six.moves import queue

fs = 44100                      # sampling frequency in Hz
td = 10                         # sweep duration in ms
NS = round(fs * (td * 1e-3))    # number of samples
fLen = 2**15                    # length of the zero padded signal

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
stream = p.open(format=pyaudio.paInt16, channels=2, rate=fs,
                input=True, frames_per_buffer=NFB, start=True,
                stream_callback=audio_callback)

fig, ax = plt.subplots(2,2)

fvals = fftfreq(fLen, 1/fs)

sb_fig = ax[0,0].plot(np.zeros(NS))[0]
Sb_fig = ax[1,0].plot(fvals[:fLen // 2], np.zeros(fLen // 2))[0]
pb_fig = ax[0,1].plot(np.zeros(NS))[0]
de_fig = ax[1,1].plot(np.arange(-gLen,0), np.zeros(gLen))[0]

#ax00.set_ylim(-1.5,1.5)
#ax1.set_title('IF spectrum')
plt.draw()
plt.pause(0.005)

data = np.zeros((2 * NS, 2))
while True:

    # capture 2 frames
    i = 0
    buff.queue.clear()
    while (i < 2 * NS):
        chunk = np.frombuffer(buff.get(), dtype=np.int16)        
        chunk = chunk.reshape((-1,2))
        chunk = chunk.astype(np.float)
        cl = chunk.shape[0]
        if (i + cl < 2 * NS):            
            cle = cl
        else:
            cle = 2 * NS - i        
        data[i:i+cle, 0] = chunk[:cle,0]
        data[i:i+cle, 1] = chunk[:cle,1]
        i += cle

    # find start and plot
    der = np.diff(data[:NS,1])
    ix = np.argmax(der)
    sb = data[ix:ix+NS,0]
    sb_fig.set_ydata(sb)
    ax[0,0].relim()
    ax[0,0].autoscale_view()

    # find FFT and plot
    Sb = np.abs(fft(sb, fLen))
    Sb = Sb / np.max(Sb)
    Sb=Sb[:fLen//2]
    Sb_fig.set_ydata(Sb)
    ax[1,0].relim()
    ax[1,0].autoscale_view()

    # find phase and plot
    pb = np.unwrap(np.angle(hilbert(sb))) * 180 / np.pi
    pb = pb - pb[0]
    pb_fig.set_ydata(pb)
    ax[0,1].relim()
    ax[0,1].autoscale_view()

    # peak log
    f_pk = fvals[np.argmax(Sb[:fLen//2])]   # * 10e-3 / 220e6 * 3e8 / 2 * 100 / 2.54 - 140
    #pkLabel.set_text("{0:.2f}Hz".format(f_pk))
    de_fig.set_ydata(np.append(de_fig.get_ydata()[-gLen + 1:], f_pk))
    ax[1,1].relim()
    ax[1,1].autoscale_view()
        
    plt.draw()
    plt.pause(0.005)
    #break

stream.stop_stream()
stream.close()
p.terminate()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, constants


t=np.linspace(0,1,1024)
chirp=signal.chirp(t,100,1,10)
plt.figure(1)
plt.title("full chirp")
plt.plot(chirp)


full_fft=np.fft.fft(chirp)
plt.figure(2)
plt.title("full chirp fft")
plt.plot(full_fft.real)
plt.plot(full_fft.imag)


p1_fft=np.fft.fft(chirp[:512])
p2_fft=np.fft.fft(chirp[512:1024])
fig,ax=plt.subplots(1, 2)
ax[0].plot(p1_fft.real)
ax[0].plot(p1_fft.imag)
ax[1].plot(p2_fft.real)
ax[1].plot(p2_fft.imag)


p1_fft+=np.random.randn(512)
p1_fft+=np.random.randn(512)*1j
p1_t=np.fft.ifft(p1_fft)
p2_t=np.fft.ifft(p2_fft)
newchirp=np.append(p1_t,p2_t)

plt.figure(4)
plt.title("reconstructed chirp transition looks continous")
plt.plot(newchirp[512-50:512+50])

plt.show()
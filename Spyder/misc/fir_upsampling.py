import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, constants, interpolate

samples=32768  # total number of upsampled datapoints
N1=4  # upsampling factor  1
N2=4  # upsampling factor  2
N3=4  # upsampling factor  3
N=N1*N2*N3  # total upsampling factor
f=samples/1024 # arbitrary freq

lo_res_samp=np.linspace(0,1,int(np.ceil(samples/N)))
lo_res_freq=np.cos(2*np.pi*f*lo_res_samp)

h1=[  -0.010259503508848203,
  -0.019941949811663168,
  -0.032739284956334144,
  -0.0427637300374936,
  -0.042978278742391074,
  -0.025378606139547837,
  0.01683334243033561,
  0.08677795767686226,
  0.18193600649908587,
  0.29341471811160075,
  0.406887534089936,
  0.5052020574616635,
  0.5720560201123595,
  0.5957489167866764,
  0.5720560201123595,
  0.5052020574616635,
  0.406887534089936,
  0.29341471811160075,
  0.18193600649908587,
  0.08677795767686226,
  0.01683334243033561,
  -0.025378606139547837,
  -0.042978278742391074,
  -0.0427637300374936,
  -0.032739284956334144,
  -0.019941949811663168,
  -0.010259503508848203]
h1=np.array(h1)


# zero stuffing
out=np.zeros(int(samples*N1/N),dtype=complex)
out[::N1]=lo_res_freq

# filter
fir_freq1=np.convolve(out,h1)
fir_freq1=fir_freq1[int(h1.size/2):-int(h1.size/2)]

# plot
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax[0].plot(fir_freq1)

ax[1].plot((np.fft.fft(fir_freq1)[:int(fir_freq1.size/2)]).real)
ax[1].set_yscale('log')




# zero stuffing
out=np.zeros(int(samples*N1*N2/N),dtype=complex)
out[::N2]=fir_freq1

# filter
fir_freq2=np.convolve(out,h1)
fir_freq2=fir_freq2[int(h1.size/2):-int(h1.size/2)]

#plt.plot(fir_freq2)

# plot
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax[0].plot(fir_freq2)

ax[1].plot(abs((np.fft.fft(fir_freq2)[:int(fir_freq2.size/2)]).real))
ax[1].set_yscale('log')



# zero stuffing
out=np.zeros(int(samples),dtype=complex)
out[::N3]=fir_freq2

# filter
fir_freq3=np.convolve(out,h1)
fir_freq3=fir_freq3[int(h1.size/2):-int(h1.size/2)]

#plt.plot(fir_freq3)
#!/usr/bin/env python3



import numpy as np
import matplotlib.pyplot as plt


def comp_CIC_kernel(N,R):
    ones=np.ones(R)
    kernel=ones
    for i in range(N-1):
        kernel=np.convolve(ones,kernel)
    
    return kernel/(R**(N-1))


def dbspec(data):
    '''Returns positive freq spectrum in dBFS (+-1). Half width of data input.'''
    
    return 20*np.log10((abs(np.fft.fft(data)[:int(len(data)/2)]))/(len(data)/2))


font = {'weight' : 'normal','size'   : 20}

plt.rc('font', **font)

fig, ax = plt.subplots(1, 2, figsize=(20, 5))

ax[0].set_title('Pulseshape')
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('Time')
ax[1].set_title('Spectral Leakage')
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Power')



ax[1].set_ylim(-150,0)

R=128
plotzeros=R*2


pulse1=comp_CIC_kernel(1,R)
pulse1=np.append(np.zeros(10001),pulse1)
pulse1=np.append(pulse1,np.zeros(10001))

ax[1].plot(dbspec(pulse1)[:1000]+37.8)


pulse_nice1=np.append(np.zeros(plotzeros),comp_CIC_kernel(1,R))
pulse_nice1=np.append(pulse_nice1,np.zeros(plotzeros))


ax[0].plot(pulse_nice1,label='first order CIC (sample and hold/square pulse)')


pulse2=comp_CIC_kernel(3,R)
pulse2=np.append(np.zeros(10000),pulse2)
pulse2=np.append(pulse2,np.zeros(10000))

ax[1].plot(dbspec(pulse2)[:1000]+37.8)

pulse_nice2=np.append(np.zeros(int(plotzeros/2)),comp_CIC_kernel(3,R))
pulse_nice2=np.append(pulse_nice2,np.zeros(int(plotzeros/2)))

ax[0].plot(pulse_nice2,label='third order CIC (cubic splines)')



pulse3=comp_CIC_kernel(6,R)
pulse3=np.append(np.zeros(10000),pulse3)
pulse3=np.append(pulse3,np.zeros(10000))

ax[1].plot(dbspec(pulse3)[:1000]+37.8)

#pulse_nice3=np.append(np.zeros(int(plotzeros/2)),comp_CIC_kernel(6,R))
#pulse_nice3=np.append(pulse_nice2,np.zeros(int(plotzeros/2)))

pulse_nice3=comp_CIC_kernel(6,R)[64:-64]

ax[0].plot(pulse_nice3,label='sixth order CIC')


ax[0].legend(loc="lower left",bbox_to_anchor=(0.0, -0.4))



#@title  { form-width: "1px" }
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:36:09 2020

@author: norman
"""
##################################################
## CIC interpolation
##
##
## Plotting of the impulse responses of different interpolations:
##
## * Sample and hold interpolation (1st order CIC)
## 
## * Cubic Splines (3rd order CIC)
## 
## * 6th order CIC
## 
##################################################
## Author: Norman Krackow
## Company: QUARTIQ GmbH
## IDE: Spyder
## Version: 2
## Date: 09.06.2020
##################################################


# %% Includes
# ================================================

import numpy as np
import matplotlib.pyplot as plt



# %% Helper functions
# ================================================

def comp_CIC_resp(order,length):
    '''Computes the CIC impulse response (fir kernel) for a given length.'''
    ones=np.ones(length)
    kernel=ones
    for i in range(order-1):
        kernel=np.convolve(ones,kernel)
    
    return kernel/(length**(order-1))


def dbspec(data):
    '''Returns positive freq spectrum in dBFS (+-1). Half width of data input.'''
    
    return 20*np.log10((abs(np.fft.fft(data)[:int(len(data)/2)]))/(len(data)/2))



# %% Parameters
# ================================================

N=100                       # CIC integrator size
max_ord=6                   # maximum order (for plotting)
extra_pad=50                # extra padding factor for better spectral resolution
lobes=5                     # approx. number of lobes to plot



# %% Compute responses and spectra
# ================================================

order=1
pulse1=np.pad(comp_CIC_resp(order,N),int((max_ord-order)*N/2))

pulse1_pad=np.pad(pulse1,extra_pad*max_ord*N)*max_ord*extra_pad

spec1=dbspec(pulse1_pad)[:int(max_ord*extra_pad*N/(lobes*2))]



order=3
pulse2=np.pad(comp_CIC_resp(order,N),int((max_ord-order)*N/2))

pulse2_pad=np.pad(pulse2,extra_pad*max_ord*N)*max_ord*extra_pad

spec2=dbspec(pulse2_pad)[:int(max_ord*extra_pad*N/(lobes*2))]



order=6
pulse3=np.pad(comp_CIC_resp(order,N),int((max_ord-order)*N/2))

pulse3_pad=np.pad(pulse3,extra_pad*max_ord*N)*max_ord*extra_pad

spec3=dbspec(pulse3_pad)[:int(max_ord*extra_pad*N/(lobes*2))]


# %% Plotting
# ================================================

plt.rc('font', size=18)
fig,ax = plt.subplots(1, 2, figsize=(15,5))

ax[0].plot(pulse1,label='order: 1')
ax[0].plot(pulse2,label='order: 3')
ax[0].plot(pulse3,label='order: 6')
ax[0].set_title('Interpolator impulse response')
ax[0].set_ylabel('Amplitude rel. full scale')
ax[0].set_xlabel('Samples')
ax[0].set_xlim(0,len(pulse1))
ax[0].legend(loc='upper right')

ax[1].plot(spec1,label='order: 1')
ax[1].plot(spec2,label='order: 3')
ax[1].plot(spec3,label='order: 6')
ax[1].set_title('Spectral leakage')
ax[1].set_ylabel('dbFS')
ax[1].set_xlabel('frequency bins')
ax[1].set_xlim(0,len(spec1))
ax[1].set_ylim(-150,0)









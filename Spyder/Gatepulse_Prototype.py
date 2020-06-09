#@title  { form-width: "1px" }
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##################################################
## Gatepulse Architecture Prototype
##
## This file is a python sketch for a pulse generation architecture
## to be deployed on the Phaser Hardware of the Sinnara ecosystem.
## Its goal is to produce pulses for single and multi qbit 
## operations on-the-fly with a certain set of parameters.
## 
##################################################
## Author: Norman Krackow
## Company: QUARTIQ GmbH
## IDE: Spyder
## Version: 4, yes...
## Date: 09.06.2020
## Status: things kinda make sense now
## Pulse: two tone around 1.21GHz
##################################################




# %% Includes
# ================================================

import numpy as np
import matplotlib.pyplot as plt


# %% Helper functions
# ================================================

def dbspec(data):
    '''Returns positive freq spectrum in dBFS (+-1). Half width of data input.'''
    
    return 20*np.log10((abs(np.fft.fft(data)[:int(len(data)/2)]))/(len(data)/2))


def match_f(f,nr_samples,rate):
    '''Returns matched freq so there are integer periods in nr_samples'''
    
    periods=round((f*nr_samples)/rate)
    return (periods*rate)/nr_samples


def upsample(data,factor):
    '''returns ideal upsampled data via zero padding in fft'''
    
    nr_zeros=int((factor-1)*len(data))       # nr zeros to pad
    data_fft=np.fft.fft(data)*factor
    data_fft_zp=np.append(data_fft[:int(len(data_fft)/2)],np.zeros(nr_zeros))
    data_fft_zp=np.append(data_fft_zp,data_fft[int(len(data_fft)/2):])
    
    return np.fft.ifft(data_fft_zp)  # upsampled sig


def add_noise(data):
    '''add 16 bit quantization noise relative to +-1'''
    
    noise_power=(2**-16)/np.sqrt(12)         # textbook magic
    return data+np.random.normal(0,noise_power,len(data))
    

# %% Plotting control
# ================================================
    
plt.rc('font', size=20)

plotting=True             # plot output setting


# %% Fundamental Parameters
# ================================================

R = 500E6                 # Sample Rate 500MHz

T = R**-1                 # Sampling period 2ns


# %% Free Parameters
# ================================================

M = 1024                  # FFT size

L = 10240                 # Total number of samples in pulse

N = 5                     # first path base upsampling factor (fft tone spacing of first path)

f_0 = 10E6                # first path DDS modulation freq

f_1 = 200E6               # DAC DDS freq

f_2 = 1E9                 # external mixer freq


# %% FFT Coefficients
# ================================================

a_m=np.zeros(M,dtype=complex)   # complex fft parameters

a_m[-10] = M/2            # (real) cosine at -10th freq with weight 1/4

a_m[10] = 1j*M/4          # (imag) sine at +10th freq with weight 1/2


# %% Pulseshape FFT Coefficients
# ================================================

p_m = np.zeros(M,dtype=complex)   # complex pulseshape fft parameters

# Hann/raised cosine Window 

p_m[0] = M/2              # DC offset

p_m[1] = -M/4              

p_m[-1] = -M/4


# %% Following Parameters
# ================================================

P = L/(M*N)               # Number of FFT blocks of first path in full pulse

assert P%1 == 0           ,"Parameter collision!"

K = L/M                   # Pulseshape upsampling factor

assert K%1 == 0           ,"Pulselength not dividable by fft size"

t_P = L*T                 # total Pulse time

print(f'Total Pulse Time: {t_P*10**6}us ({L} Samples)')

# Match mixing frequency to fit into the samples with integer multiples of whole periods.
# This avoids weird leakage artifacts when plotting the spectra.
# Not necessary in real implementation.

f_0_m = match_f(f_0,L,R)  # first path DDS modulation freq matched


# compute and print frequency components

nr_freqs=0                # number of frequencys in first path

tone_spacing=R/(N*M)      # tone spacing of first path

f_c=f_0+f_1+f_2           # final carrier after 3x upconversion  

tone=[]                   # tones
for i,e in enumerate(a_m):
    
    if(e!=0):
        if i<=512:
            list.append(tone,i*tone_spacing)
            print(f'Tone at {f_c} + {i*tone_spacing} with parameter {e}')
        else:
            list.append(tone,-(M-i)*tone_spacing)
            print(f'Tone at {f_c} - {(M-i)*tone_spacing} with parameter {e}')
            
        nr_freqs=nr_freqs+1


# %% STFT, gating, upsampling
# ================================================

STFT_block=np.fft.fft(a_m)/M
sig_gated=STFT_block
for i in range(int(P-1)):
    sig_gated=np.append(sig_gated,STFT_block)

sig_us=upsample(sig_gated,N)



# %% Generate carrier and modulate
# ================================================

pulse_time=np.linspace(0,t_P-T,L)# timeseries for the whole pulse

carrier_0=np.exp(2j*np.pi*f_0_m*pulse_time)

pulse_modu=sig_us*carrier_0   # modulated pulse complex


# %% Generate pulseshape and shape
# ================================================

shape=np.fft.ifft(p_m).real     # base shape

shape_us=upsample(shape,K).real # upsampled shape

pulse=pulse_modu*shape_us       # final pulse before DAC

pulse=add_noise(pulse)          # add awgn

pulse_spec=dbspec(pulse)        # pulse spectrum


# %% Plotting
# ================================================

if plotting:
    
    fig,ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].plot(STFT_block.real,label='Inphase')
    ax[0].plot(STFT_block.imag,label='Quadrature')
    ax[0].set_title('Inphase/Quadrature STFT output')
    ax[0].set_ylabel('Amplitude rel. full scale')
    ax[0].set_xlabel('Samples')
    ax[0].set_xlim(0,M)
    ax[0].legend(loc='upper right')
    
    ax[1].plot(shape)
    ax[1].set_title('Pulseshape')
    ax[1].set_ylabel('Amplitude rel. full scale')
    ax[1].set_xlabel('Samples')
    ax[1].set_xlim(0,M)
    
    
    fig,ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(pulse_time*10**6,pulse.real)
    ax.set_title('Inphase pulse before upconversion')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (us)')
    ax.set_xlim(0,t_P*10**6)
    
    # virtually shift Pulse up to f_1+f_2 --> tones at f_0+f_1+f_2+-((m/M)*R/N)
    
    f_low=(f_1+f_2)*10**-9      # upper bound of freq spectrum
    f_up=f_low+(R/2)*10**-9     # lower bound of freq spectrum
    pulse_freqs=np.linspace(f_low,f_up,int(L/2))
    
    fig,ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(pulse_freqs,pulse_spec)
    ax.set_title('Upconverted pulse spectrum')
    ax.set_ylabel('dBFS')
    ax.set_xlabel('Freq (GHz)')
    ax.set_xlim(1.2,1.24)
    ax.set_ylim(-160,0)
    ax.grid(True, which='both')
    ax.minorticks_on()
    ax.annotate(f'$f_c={f_c*10**-9}GHz$ \n$f_r=f_c{tone[1]*10**-3}kHz$ \n$f_b=f_c+{tone[0]*10**-3}kHz$',
                xy=(0.78, 0.7), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='w'))



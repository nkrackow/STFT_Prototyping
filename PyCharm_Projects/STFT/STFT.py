
#   Python prototyping of the Short Time Furier Transform based Pulse generation Engine
#   for Phaser Hardware





#imports

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, constants


# STFT Processor testing

class STFT:
    """Short Time Furier Transform based Pulse generation Engine"""

    def __init__(self, nr_ffts=4, fft_points=256, pulse_samples=2048):
        self.nr_ffts=nr_ffts
        self.blocks=int(pulse_samples/fft_points)
        assert not(pulse_samples%fft_points)
        self.fft_points=fft_points
        self.pulse_samples=pulse_samples

    def Pulse(self, coeff=np.zeros(256), alpha=0.5):
        samples=np.fft.ifft(coeff)
        pulse=np.zeros(0)
        for b in range(self.blocks):
            pulse=np.append(pulse,samples)

        plt.figure(1)
        plt.plot(np.real(pulse))
        plt.figure(2)
        plt.plot(np.absolute(np.fft.fft(pulse)))
        plt.show()



if __name__ == "__main__":
    coeff=np.zeros(256)
    #coeff[0]=128
    #coeff[-1]=128
    coeff[2]=128
    coeff[-2]=128
    envelope=(np.cos(np.linspace(0,np.pi,256))+1)*0.5
    t_temp = np.fft.ifft(coeff)
    plt.figure(1)
    plt.plot(np.real(t_temp))
    plt.plot(np.imag(t_temp))
    comb=envelope*t_temp
    spec=np.fft.fft(comb)
    plt.figure(2)
    plt.plot(np.real(spec))
    plt.plot(np.imag(spec))
    spec = np.fft.ifft((spec))
    plt.figure(3)
    plt.plot(np.real(spec))
    plt.plot(np.imag(spec))

    plt.show()


    
    processor=STFT()
    processor.Pulse(coeff)
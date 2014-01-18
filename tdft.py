'''
Time Dependent Fourier Transform (Spectrogram)
Bryant Moquist
'''

import scipy
import numpy as np

def tdft(audio, srate, windowsize, windowshift,fftsize):

    """Calculate the real valued fast Fourier transform of a segment of audio multiplied by a 
    a Hamming window.  Then, convert to decibels by multiplying by 20*log10.  Repeat for all
    segments of the audio."""
    
    windowsamp = int(windowsize*srate)
    shift = int(windowshift*srate)
    window = scipy.hamming(windowsamp)
    spectrogram = scipy.array([20*scipy.log10(abs(np.fft.rfft(window*audio[i:i+windowsamp],fftsize))) 
                     for i in range(0, len(audio)-windowsamp, shift)])
    return spectrogram
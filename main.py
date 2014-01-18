''' 
Music Identification Program (a.k.a. Shazam/Soundhound) 
Proof of Concept 
Bryant Moquist
'''
from __future__ import print_function
import scipy, pylab
from scipy.io.wavfile import read
import sys
import peakpicker as pp
import fingerprint as fhash
import matplotlib
import numpy as np
import tdft

if __name__ == '__main__':

    num_inputs = len(sys.argv)

    #Song files to be hashed into database
    songs = []
    songnames = []
    separator = '.'

    for i in range(1,num_inputs):
    	songs.append(read(sys.argv[i]))
    	name = sys.argv[i].split(separator,1)[0]
    	songnames.append(name)

    #TDFT parameters
    windowsize = 0.008     #set the window size  (0.008s = 64 samples)
    windowshift = 0.004    #set the window shift (0.004s = 32 samples)
    fftsize = 1024         #set the fft size (if srate = 8000, 1024 --> 513 freq. bins separated by 7.797 Hz from 0 to 4000Hz) 
    
    #Peak picking dimensions 
    f_dim1 = 30
    t_dim1 = 80 
    f_dim2 = 10
    t_dim2 = 20
    percentile = 70
    base = 70 # lowest frequency bin used (peaks below are too common/not as useful for identification)
    high_peak_threshold = 75
    low_peak_threshold = 60

    #Hash parameters
    delay_time = 250      # 250*0.004 = 1 second
    delta_time = 250*3    # 750*0.004 = 3 seconds
    delta_freq = 128      # 128*7.797Hz = approx 1000Hz
    
    #Time pair parameters
    TPdelta_freq = 4
    TPdelta_time = 2

    #Construct the audio database of hashes
    database = np.zeros((1,5))
    spectrodata = []
    peaksdata = []

    for i in range(0,len(songs)):

    	print('Analyzing '+str(songnames[i]))
    	srate = songs[i][0]  #sample rate in samples/second
    	audio = songs[i][1]  #audio data    	
    	spectrogram = tdft.tdft(audio, srate, windowsize, windowshift, fftsize)
    	time = spectrogram.shape[0]
    	freq = spectrogram.shape[1]

    	threshold = pp.find_thres(spectrogram, percentile, base)

    	print('The size of the spectrogram is time: '+str(time)+' and freq: '+str(freq))
    	spectrodata.append(spectrogram)

    	peaks = pp.peak_pick(spectrogram,f_dim1,t_dim1,f_dim2,t_dim2,threshold,base)

    	print('The initial number of peaks is:'+str(len(peaks)))
    	peaks = pp.reduce_peaks(peaks, fftsize, high_peak_threshold, low_peak_threshold)

    	print('The reduced number of peaks is:'+str(len(peaks)))
    	peaksdata.append(peaks)

    	#Calculate the hashMatrix for the database song file
    	songid = i
    	hashMatrix = fhash.hashPeaks(peaks,songid,delay_time,delta_time,delta_freq)

        #Add to the song hash matrix to the database
    	database = np.concatenate((database,hashMatrix),axis=0)


    print('The dimensions of the database hash matrix: '+str(database.shape))
    database = database[np.lexsort((database[:,2],database[:,1],database[:,0]))]

    # Audio sample to be analyzed and identified

    print('Please enter an audio sample file to identify: ')
    userinput = raw_input('---> ')
    sample = read(userinput)
    userinput = userinput.split(separator,1)[0]

    print('Analyzing the audio sample: '+str(userinput))
    srate = sample[0]  #sample rate in samples/second
    audio = sample[1]  #audio data    	
    spectrogram = tdft.tdft(audio, srate, windowsize, windowshift, fftsize)
    time = spectrogram.shape[0]
    freq = spectrogram.shape[1]
    
    print('The size of the spectrogram is time: '+str(time)+' and freq: '+str(freq))

    threshold = pp.find_thres(spectrogram, percentile, base)
    
    peaks = pp.peak_pick(spectrogram,f_dim1,t_dim1,f_dim2,t_dim2,threshold,base)
    
    print('The initial number of peaks is:'+str(len(peaks)))
    peaks = pp.reduce_peaks(peaks, fftsize, high_peak_threshold, low_peak_threshold)
    print('The reduced number of peaks is:'+str(len(peaks)))

    #Store information for the spectrogram graph
    samplePeaks = peaks
    sampleSpectro = spectrogram

    hashSample = fhash.hashSamplePeaks(peaks,delay_time,delta_time,delta_freq)
    print('The dimensions of the hash matrix of the sample: '+str(hashSample.shape))

    print('Attempting to identify the sample audio clip.')
    timepairs = fhash.findTimePairs(database, hashSample, TPdelta_freq, TPdelta_time)

    #Compute number of matches by song id to determine a match
    numSongs = len(songs)
    songbins= np.zeros(numSongs)
    numOffsets = len(timepairs)
    offsets = np.zeros(numOffsets)
    index = 0
    for i in timepairs:
    	offsets[index]=i[0]-i[1]
    	index = index+1
    	songbins[i[2]] += 1

    # Identify the song
    print('The sample song is: '+str(songnames[np.argmax(songbins)]))

    # Plots 
    fig = []

    # Plot the magnitude spectrograms
    for i in range(0,numSongs):
        fig1 = pylab.figure(i)
        peaks = peaksdata[i]
        pylab.imshow(spectrodata[i].T,origin='lower', aspect='auto', interpolation='nearest')
        pylab.scatter(*zip(*peaks), marker='.', color='blue')
        pylab.title(str(songnames[i])+' Spectrogram and Selected Peaks') 
        pylab.xlabel('Time')    
        pylab.ylabel('Frequency Bin')
        fig.append(fig1)
    
    #Show the figures
    for i in fig:
        i.show()

    fig2 = pylab.figure(1002)
    pylab.imshow(sampleSpectro.T,origin='lower', aspect='auto', interpolation='nearest')
    pylab.scatter(*zip(*samplePeaks), marker='.', color='blue')
    pylab.title('Sample File: '+str(userinput)+' Spectrogram and Selected Peaks') 
    pylab.xlabel('Time')    
    pylab.ylabel('Frequency Bin')
    fig2.show()

    fig3 = pylab.figure(1003)
    ax = fig3.add_subplot(111)

    ind = np.arange(numSongs)
    width = 0.35
    rects1 = ax.bar(ind,songbins,width,color='blue',align='center')
    ax.set_ylabel('Number of Matches')
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(songnames)
    matplotlib.pyplot.setp(xtickNames)
    pylab.title('Song Identification') 
    fig3.show()

    pylab.show()

    print('The sample song is: '+str(songnames[np.argmax(songbins)]))


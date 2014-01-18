'''
Peak Picking Functions
Bryant Moquist
'''
import numpy as np

def find_thres(spectrogram,percentile,base):
    "Find the peak picking threshold for a particular spectrogram"
    dim = spectrogram.shape
    window = spectrogram[0:dim[0],base:dim[1]]
    threshold = np.percentile(window, percentile)

    return threshold

def peak_pick (S,f_dim1,t_dim1,f_dim2,t_dim2,threshold,base):
    "Selects local peaks in a spectrogram and returns a list of tuples (time, freq, amplitude)" 
    "S is spectrogram matrix"
    "f_dim1,f_dim2,t_dim1,and t_dim2 are freq x time dimensions of the sliding window for first and second passes"
    "threshold is the minimum amplitude required to be a peak"
    "base is the lowest frequency bin considered"

    a = len(S) #num of time bins
    b = len(S[1]) #num of frequency bins

    peaks = []
    t_coords = []
    f_coords = []

    "Determine the time x frequency window to analyze"
    for i in range(0,a,t_dim1):
        for j in range(base,b,f_dim1):
            if i + t_dim1 < a and j + f_dim1 < b:
                window = S[i:i+t_dim1,j:j+f_dim1]
            elif i + t_dim1 < a and j + f_dim1 >= b:
                window = S[i:i+t_dim1,j:b]
            elif i + t_dim1 >= a and j + f_dim1 < b:
                window = S[i:a,j:j+f_dim1]
            else:
                window = S[i:a,j:b]

            "Check if the largest value in the window is greater than the threshold"
            if np.amax(window) >= threshold:
                row, col = np.unravel_index(np.argmax(window), window.shape) # pulls coordinates of max value from window
                t_coords.append(i+row)
                f_coords.append(j+col) 
     
    "Iterates through coordinates selected above to make sure that each of those points is in fact a local peak"    
    for k in range(0,len(f_coords)):
        fmin = f_coords[k] - f_dim2
        fmax = f_coords[k] + f_dim2
        tmin = t_coords[k] - t_dim2
        tmax = t_coords[k] + t_dim2
        if fmin < base:
            fmin = base
        if fmax > b:
            fmax = b
        if tmin < 0:
            tmin = 0
        if tmax > a:
            tmax = a
        window = S[tmin:tmax,fmin:fmax] #window centered around current coordinate pair

        "Break when the window is empty"
        if not window.size:
            continue

        "Eliminates coordinates that are not local peaks by setting their coordinates to -1"
        if S[t_coords[k],f_coords[k]] < np.amax(window):
            t_coords[k] = -1
            f_coords[k] = -1

    "Removes all -1 coordinate pairs"
    f_coords[:] = (value for value in f_coords if value != -1)
    t_coords[:] = (value for value in t_coords if value != -1)

    for x in range(0, len(f_coords)):
        peaks.append((t_coords[x], f_coords[x], S[t_coords[x], f_coords[x]]))

    return peaks

def reduce_peaks(peaks,fftsize,high_peak_threshold,low_peak_threshold):
    "Reduce the number of peaks by separating into high and low frequency regions and thresholding."
    #Separate regions ensure better spread of peaks. 
    low_peaks = []
    high_peaks = []

    for item in peaks:
        if(item[1]>(fftsize/4)):
            high_peaks.append(item)
        else:
            low_peaks.append(item)
    
    #Eliminate peaks based on respective thresholds in the low and high frequency regions.  
    reduced_peaks = []
    for item in peaks:
        if(item[1]>(fftsize/4)):
            if(item[2]>np.percentile(high_peaks,high_peak_threshold,axis=0)[2]):
                reduced_peaks.append(item)
            else:
                continue
        else:
            if(item[2]>np.percentile(low_peaks,low_peak_threshold,axis=0)[2]):
                reduced_peaks.append(item)
            else:
                continue

    return reduced_peaks
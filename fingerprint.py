'''
Hash and Acoustic Fingerprint Functions
Bryant Moquist
'''

import numpy as np

def findAdjPts(index,A,delay_time,delta_time,delta_freq):
    "Find the three closest adjacent points to the anchor point"    
    adjPts = []
    low_x = A[index][0]+delay_time
    high_x = low_x+delta_time
    low_y = A[index][1]-delta_freq/2
    high_y = A[index][1]+delta_freq/2
    
    for i in A:
        if ((i[0]>low_x and i[0]<high_x) and (i[1]>low_y and i[1]<high_y)):
            adjPts.append(i)
            
    return adjPts
    
def hashPeaks(A,songID,delay_time,delta_time,delta_freq):
    "Create a matrix of peaks hashed as: [[freq_anchor, freq_other, delta_time], time_anchor, songID]"
    hashMatrix = np.zeros((len(A)*100,5))  #Assume size limitation
    index = 0
    numPeaks = len(A)
    for i in range(0,numPeaks):
        adjPts = findAdjPts(i,A,delay_time,delta_time,delta_freq)
        adjNum=len(adjPts)
        for j in range(0,adjNum):
            hashMatrix[index][0] = A[i][1]
            hashMatrix[index][1] = adjPts[j][1]
            hashMatrix[index][2] = adjPts[j][0]-A[i][0]
            hashMatrix[index][3] = A[i][0]
            hashMatrix[index][4] = songID
            index=index+1
    
    hashMatrix = hashMatrix[~np.all(hashMatrix==0,axis=1)]
    hashMatrix = np.sort(hashMatrix,axis=0)
        
    return hashMatrix

def hashSamplePeaks(A,delay_time,delta_time,delta_freq):
    "Create a matrix of peaks hashed as: [[freq_anchor, freq_other, delta_time],time_anchor]"
    hashMatrix = np.zeros((len(A)*100,4))
    index = 0
    numPeaks = len(A)
    for i in range(0,numPeaks):
        adjPts = findAdjPts(i,A,delay_time,delta_time,delta_freq)
        adjNum = len(adjPts)
        for j in range(0,adjNum):
            hashMatrix[index][0] = A[i][1]
            hashMatrix[index][1] = adjPts[j][1]
            hashMatrix[index][2] = adjPts[j][0]-A[i][0]
            hashMatrix[index][3] = A[i][0]
            index=index+1

    hashMatrix = hashMatrix[~np.all(hashMatrix==0,axis=1)]
    hashMatrix = np.sort(hashMatrix,axis=0)
        
    return hashMatrix

def findTimePairs(hash_database,sample_hash,deltaTime,deltaFreq):
    "Find the matching pairs between sample audio file and the songs in the database"

    timePairs = []

    for i in sample_hash:
        for j in hash_database:
            if(i[0] > (j[0]-deltaFreq) and i[0] < (j[0] + deltaFreq)):
                if(i[1] > (j[1]-deltaFreq) and i[1] < (j[1] + deltaFreq)):
                    if(i[2] > (j[2]-deltaTime) and i[2] < (j[2] + deltaTime)):
                        timePairs.append((j[3],i[3],j[4]))
                    else:
                        continue
                else:
                    continue
            else:
                continue
            
    return timePairs
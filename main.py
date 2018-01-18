from scipy.io import wavfile as wav
from scipy.fftpack import fft as fft_external
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import medfilt
from scipy.signal import find_peaks_cwt
from scipy.signal import decimate
from cmath import exp, pi


def dft(signal):
    n = len(signal)
    result = [0] * len(signal)

    for q in range(n):
        for w in range(n):
            result[q] += signal[w] * math.e**((-1) * 2j * math.pi * q * w / n )
    return result

def fft(signal):
    N = len(signal)

    if N <= 1: 
        return signal

    evenValues = fft(signal[::2])
    oddValues =  fft(signal[1::2])

    T = [exp(-2j * pi * k / N) * oddValues[k] for k in range(N//2)]
    [evenValues[k] for k in range(N//2)]
    res = ([evenValues[k] + T[k] for k in range(N//2)] +
           [evenValues[k] - T[k] for k in range(N//2)])
    return res
        


def nextPowerOf2(x):  
    return 2**(x-1).bit_length()


def hps(dftData, downsamplingFactor):
    
    spectrum = list(dftData[0:len(dftData)//2])

    # Calculate amplitudes
    for i in range(len(spectrum)):
        spectrum[i] = math.sqrt((spectrum[i].real ** 2) + (spectrum[i].imag * spectrum[i].imag))

    signalAfterHPS = list(spectrum)

    for i in range(1, downsamplingFactor + 1):
        for index in range(len(spectrum) // i):
            signalAfterHPS[index] *= spectrum[index * i]
    
    return signalAfterHPS


def getAmplitudeVector(dftData):

    ampVector = [0] * len(dftData)
    for i in range(len(dftData)):
        ampVector[i] = math.sqrt(dftData[i].real ** 2 + dftData[i].imag ** 2)
    
    return ampVector


def getFundamentalFrequency(xFrequencies, amplitudeVector):
    maxAmplitude = float('-inf')
    indexMaxAmplitude = -1

    for i in range(1, len(amplitudeVector)):
        if maxAmplitude < amplitudeVector[i]:
            maxAmplitude = amplitudeVector[i]
            indexMaxAmplitude = i

    return xFrequencies[indexMaxAmplitude]

# Notes
# C4 - 261.626
# C5 - 523.251
# C6 - 1046.50
# G3 - 195.998
# G4 - 391.995
# G5 - 783.991
# G6 - 1567.98
def getNote(frequency):
    range = 20

    if frequency > 261.626 - range and frequency < 261.626 + range:
        return 'C4'
    elif frequency > 523.251 - range and frequency < 523.251 + range:
        return 'C5'
    elif frequency > 1046.50 - range and frequency < 1046.50 + range:
        return 'C6'
    elif frequency > 195.998 - range and frequency < 195.998 + range:
        return 'G3'
    elif frequency > 391.995 - range and frequency < 391.995 + range:
        return 'G4'
    elif frequency > 783.991 - range and frequency < 783.991 + range:
        return 'G5'
    elif frequency > 1567.98 - range and frequency < 1567.98 + range:
        return 'G6'

if __name__ == "__main__":

    rate, signal = wav.read('notes/piano-G4.wav')
    # rate, signal = wav.read('C5-G5-G6-C4.wav')
    # rate, signal = wav.read('C5-C4-C4-G3-G4-G6-C5.wav')
    signal = np.asfarray(signal, float)

    plt.plot(signal)
    plt.show()

    # normalize the signal
    maxRange = max(max(signal), abs(min(signal)))

    for i in range(len(signal)):
        signal[i] = signal[i] / maxRange

    # Calculate the window averages
    windowSize = 1500
    absAverage = []
    percentageIncreaseForNoteDetected = 1.5
    minNoteAvgThreshold = 0.1

    for i in range(0,len(signal), windowSize):
        absAverage.append(sum(map(abs, signal[i:(i+windowSize)])) / len(signal[i:(i+windowSize)]))

    # Detect a new note by finding large changes in the averages
    noteStartingIndexes = []
    minNoteAbsAverage = float('inf')

    for i in range(len(absAverage) - 1):
        if absAverage[i+1] / absAverage[i] > percentageIncreaseForNoteDetected :
            noteStartingIndexes.append((i+1) * windowSize)
        
            if absAverage[i+1] < minNoteAbsAverage:
                minNoteAbsAverage = absAverage[i+1]

    # Check if first window contains a note
    if absAverage[0] > minNoteAbsAverage or absAverage[0] > minNoteAvgThreshold:
        noteStartingIndexes.insert(0, 0)

    # Identify the note in each window
    for i in range(len(noteStartingIndexes)):
        partialSignal = []

        if i == len(noteStartingIndexes) - 1:
            partialSignal = signal[noteStartingIndexes[i]:]
        else:
            partialSignal = signal[noteStartingIndexes[i]:noteStartingIndexes[i+1]]
        
        # Add padding to get to power of 2
        if 2**(len(partialSignal).bit_length() -1) != len(partialSignal):
            padding = [0] * (nextPowerOf2(len(partialSignal)) - len(partialSignal))
            partialSignal = np.append(partialSignal, padding)
        
        dft_data = fft(partialSignal)
        hpsResult = hps(dft_data, 2)

        amplitudeVector = getAmplitudeVector(hpsResult)

        # Get the frequencies for the x-axis
        xFrequencies = [0] * (len(partialSignal) // 2)
        for j in range(len(xFrequencies)):
            xFrequencies[j] = j * rate / len(partialSignal)

        # Print the fundamental frequency (maximum frequency after applying Harmonic Product Spectrum on the FFT result )
        fundamentalFrequency = getFundamentalFrequency(xFrequencies, amplitudeVector)

        print('{} ({})'.format(getNote(fundamentalFrequency), fundamentalFrequency))
        # print('The fundamental frequency is {} ({}):'.format(getNote()), fundamentalFrequency)

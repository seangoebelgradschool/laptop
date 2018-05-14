import numpy as np
import matplotlib.pyplot as plt
import pdb

def test():
    arr = np.random.normal(size=2**12) + \
          np.sin(np.arange(2**12)/10.*2.*np.pi)
    
    arr -= np.median(arr)

    arr[np.arange(len(arr)/6)*6] = 0
    arr[np.arange(len(arr)/6)*6+1] = 0

    fft = np.fft.fft(arr)
    powspec = (abs(fft))**2
    readout_rate = 100. #fps
    xaxis = np.fft.fftfreq(powspec.size, d=1./readout_rate)
    
    plt.plot(arr)
    plt.show()

    #plt.figure(1, figsize=(15, 5), dpi=100)

    plt.subplot(2,1,1)
    plt.plot(xaxis, powspec)
    #plt.yscale('log')
    plt.xlim((-1, 50))
    plt.xlabel("Hz")
    plt.title("Zeros")

    #interpolate the missing points
    for i in range(len(arr)):
        if (arr[i] == 0) and (i>1):
            arr[i]   = (2*arr[i-1] + arr[i+2])/3
            arr[i+1] = (arr[i-2] + 2*arr[i+1])/3
    
    fft = np.fft.fft(arr)
    powspec = (abs(fft))**2

    plt.subplot(2,1,2)
    plt.plot(xaxis, powspec)
    plt.title("Interpolated")
    #plt.yscale('log')
    plt.xlim((-1, 50))
    plt.xlabel("Hz")

    plt.show()

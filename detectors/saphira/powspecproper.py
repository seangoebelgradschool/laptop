#started on 2017-04-24
#plots power spectrum of RFI, taking into account bonus clock cycles
#The latest version is on SCExAO.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.stats import mode
import os
#import time

#Declare some global variables
dir = 'images/'
filename = 'saphira_14:46:28.625120914.fits'

t_pix = 1e-6 #seconds per pixel
t_endrow = 540e-9 #seconds at the end of each row
t_endframe = 310e-9 #seconds at the end of each frame
t_clock = 10e-9 #duration of clock cycle
#end of row clock at end of frame?

print "Reading image..."
#img = pyfits.getdata(dir+filename)[10:1000]
#img = np.random.normal(loc=1e4, scale=1000, size=(10, 256, 320))
img = np.random.uniform(low=100, high=200, size=(10, 256, 320))
img_avg = np.median(img, 0)
for z in range(np.shape(img)[0]):
    img[z] -= img_avg
print "Image read and median subtracted. Size is ", np.shape(img)
print "Median, stddev:", np.median(img), np.std(img, ddof=1)

def main():
    row_clocks = np.shape(img)[2]/32*t_pix/t_clock + t_endrow/t_clock
    pixel_arr = np.zeros(((row_clocks * np.shape(img)[1] + t_endframe/t_clock) * \
                          np.shape(img)[0] ) )#+500#.astype(int)

    for z in range(np.shape(img)[0]):
        if z % (np.shape(img)[0] /100.) < np.shape(img)[0]/100.: #update progress
            print str(int(round(float(z) / np.shape(img)[0] * 100.)))+ "% complete."
        for y in range(np.shape(img)[1]):
            for x in range(np.shape(img)[2]/32):
                pixel_arr[z*(row_clocks*np.shape(img)[1] + t_endframe/t_clock) + \
                          y*row_clocks +     x*t_pix/t_clock : \
                          z*(row_clocks*np.shape(img)[1] + t_endframe/t_clock) + \
                          y*row_clocks + (x+1)*t_pix/t_clock] \
                    = np.median(img[z,y,x*32:(x+1)*32])

    print "Computing fft."
    fft = np.fft.fft(pixel_arr)
    xaxis = np.fft.fftfreq(pixel_arr.size, d=t_clock)
    powspec = (abs(fft))**2

    plt.plot(xaxis, powspec)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim((np.shape(img)[0]*(row_clocks*np.shape(img)[1] + \
                                t_endframe/t_clock)*t_clock)**(-1) , \
             0.5*t_pix**(-1))
    plt.title('Power Spectrum '+filename + ' ' +str(np.shape(img)))
    plt.ylabel('Power (arb. units)')
    plt.xlabel('Frequency (Hz)')
    plt.show()




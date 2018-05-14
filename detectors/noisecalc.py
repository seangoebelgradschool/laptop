#/usr/bin/env python
#Updated 4/11/14 4:20 PM
#LOCAL COPY

# Plots the variance of varying CDS frame deltas in order to determine the shot
# and read noise.

import pyfits
import matplotlib.pyplot as plt
import numpy as np

#dir='/home/H4RG/Data/SpeX_LINEARITY_TESTING/test17_gain_low_bias_0.500V_3usec/'
#filename_sub = 'buzz2-00345.Z.'
dir='h4rg_illum/'
filename_sub = 'H4RG_R01_M01_N'

files = np.arange(10, 40, 1) #file list to loop over
#f9 = pyfits.getdata(dir+filename_sub+'9.fits') #For H2RG
f9 = pyfits.getdata(dir+filename_sub+'09.fits') #FOr H4RG

n_crops = 7 #Number of crops to check, should be integer
center = ((f9.shape)[1])/2.
yshifts = np.zeros(n_crops)

#display crops for user to decide if they're good
for i in range(n_crops):
    answer = 'n'
    yshift = 0
    while (answer != 'y') :
        frame = pyfits.getdata(dir+filename_sub+'39.fits')
        crop  = (frame-f9)[center+yshift : center+70+yshift , 
                           center+(i-n_crops/2)*64 : center+(i-n_crops/2)*64 +64]
        
        plt.imshow(crop, interpolation='nearest', vmin=np.median(crop)-100,
                   vmax=np.median(crop)+100)
        plt.colorbar()
        plt.show()
        
        #ask user if the crop is acceptable
        answer = raw_input('Is the crop clean (y/n)? ')
        print
        if (answer.lower() != 'y'): yshift += 70
        yshifts[i] = yshift
        

noisearr = np.zeros((len(files) , n_crops)) #stores variances
readnoise = np.zeros(n_crops) #stores calculated read noises

for i in range(len(files)): #range(len(files)): #cds frame delta
    frame = pyfits.getdata(dir+filename_sub+str(files[i])+'.fits')
    cds  = frame-f9
    print str(int(round(float(i)/len(files)*100.))), '% done.'

    for j in range(n_crops):
        #crop = cds[1024:1216 , 800:1080] #clean area of H2RG
        crop = cds[center + yshifts[j]       : center + yshifts[j] + 70 , 
                   center + (j-n_crops/2)*64 : center+(j-n_crops/2)*64 + 64]

        #SUBTRACT REFERENCE PIXELS
        #...
    
        #calculate mask with bad pixels in place
        mask = np.where( (crop > np.median(crop) - 3.*np.std(crop, ddof=1)) & 
                         (crop < np.median(crop) + 3.*np.std(crop, ddof=1)) )
        #rinse and repeat without bad pixels
        mask = np.where( (crop > np.median(crop[mask]) - 3.*np.std(crop[mask], ddof=1)) & 
                         (crop < np.median(crop[mask]) + 3.*np.std(crop[mask], ddof=1)) )

        #check masking periodically
        if (files[i] % 70 == 0):
            junk = plt.hist(crop.flatten(), bins=1000, log=True, range=[0,2000])
            plt.title("Pixel Brightness Histogram, CDS "+str(files[i])+'-09')
            plt.xlabel('ADU')
            plt.ylabel('N_Occurences')
            plt.vlines( (np.min(crop[mask]) , np.max(crop[mask])) , 0.1, 1e4, colors='r')
            plt.show()
            
        noisearr[i, j] = np.var(crop[mask], ddof=1)

for i in range((noisearr.shape)[1]):
    plt.plot(files-9, noisearr[:,i], 'o')
    plt.title('Pixel Variance vs. CDS Frame Delta')
    plt.xlabel('CDS Frame Delta')
    plt.ylabel('Variance [ADU^2]')
        
    #fit a polynomial to the data
    coeffs = np.polyfit(files-9, noisearr[:,i], 1)
    fit = np.poly1d(coeffs)
    ys = fit(files-9)
        
    plt.plot(files-9, ys)
    plt.text(1, np.max(noisearr[:,i]), 'y='+str(coeffs[0])[0:4]+'x+'+str(coeffs[1])[0:4])
    plt.show()
        
    #store read noise (y-intercept) value in larger array
    readnoise[i] = coeffs[1]

readnoise = np.sqrt(readnoise) #convert to ADU

print "Read noise values [ADU]:", readnoise
print "Mean [ADU]:", np.mean(readnoise)
print "Median [ADU]:",  np.median(readnoise)
print "Uncertainty (sigma/root(N)) [ADU]:", np.std(readnoise, ddof=1) / np.sqrt(n_crops)

#/usr/bin/env python

#updated 12/5/15
#Reads decubed/aligned images. Subtracts adjacent images to see
# how long it takes for speckles to change by ___ fraction of core
# flux.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation
import pdb

def speckles():
    dir = 'images/'
    filename = '150729_230025_cleanedaligned.fits'
    img = pyfits.getdata(dir+filename)#[0:5000]

    img_avg = np.sum(img,0) / np.shape(img)[0]
    
    selection = img_avg > np.sort(img_avg.flatten())[np.size(img_avg)-30] #29 brightest pixels
    x0 = np.sum(np.sum(img_avg*selection, 0)*range(np.shape(img_avg)[1])) / \
         np.sum(img_avg*selection)
    y0 = np.sum(np.sum(img_avg*selection, 1)*range(np.shape(img_avg)[0])) / \
         np.sum(img_avg*selection)

    #too lazy to code efficiently
    r_max = 30. #pix distance from PSF core
    r_min = 5.
    keeperpix = np.zeros(np.shape(img_avg))
    keeperpix[:] = False
    for x in range(np.shape(keeperpix)[1]):
        for y in range(np.shape(keeperpix)[0]):
            if ((x-x0)**2 + (y-y0)**2 < r_max**2) & ((x-x0)**2 + (y-y0)**2 > r_min**2):
                keeperpix[y,x] = True
    mask = np.where(keeperpix == True)

    if 0:
        plt.figure(1, figsize=(15, 5), dpi=100) 
        plt.subplot(131)
        plt.imshow(keeperpix, interpolation='none')

        plt.subplot(132)
        plt.imshow(np.log(img_avg), interpolation='none')
        
        plt.subplot(133)
        plt.imshow(np.log(keeperpix*img_avg), interpolation='none')
        
        plt.show()


    separations = np.arange(50)+1
    stddev_storage = np.zeros(np.size(separations))

    #how many times to repeat stddev measurement
    iterations = np.round(np.linspace(0, np.shape(img)[0]-np.max(separations)-1, 500))

    for i in iterations: #last number is how many iterations
        if i%(len(iterations)/100)==0: #update status
            print str(int(round(float(i)/np.max(iterations)*100.)))+"% complete."

        #loc = np.where(keeperpix == True)
        #threshold = img[i][loc] < np.sort(img[i][loc].flatten())[0.5*np.sum(keeperpix)]
        loc = np.where(img[i][mask] < np.sort(img[i][mask])[0.5*np.sum(keeperpix)])
        for sep in separations:
            ccds = img[i] - img[i+sep]
            stddev_storage[sep-1] += np.std(ccds[mask][loc], ddof=1)
    stddev_storage /= float(len(iterations))
        
    #n_avgs = 0
    #stdev = 0
    #stdevs = np.array([])
    #separations = np.arange(50)+1
    #for i in range(500):
    #    print str(int(round(float(sep)/np.max(separations)*100.)))+"% complete."
    #    for sep in separations:
    #        pdb.set_trace()
    #        ccds = (img[i] - img[i+sep])*keeperpix
    #        selection = np.where(ccds < np.sort(ccds.flatten())*0.5*np.size(ccds))
    #        stdev += np.std(ccds*selection, ddof=1)
    #        n_avgs += 1

        #need to store averages!
        #stdevs = np.append(stdevs, stdev/n_avgs)

    #fit a polynomial to the variance data
    #coeffs = np.polyfit(separations, stdevs, 2) #quadratic fit
    #fit = np.poly1d(coeffs)
    #noisefit = fit(separations)

    psf_core = np.max(img_avg)
    
    plt.plot(separations, stddev_storage/psf_core, 'o')
    #plt.plot(separations, noisefit)
    #plt.text(1, 0.8*np.max(stdevs), 'y='+str(coeffs[0])[0:6]+'x^2+'+
    #         str(coeffs[1])[0:4]+'x+' + str(coeffs[2])[0:4])

    plt.title(filename[:13]+', PSF Core = '+str(int(round(psf_core)))+' ADU')
    plt.xlabel('Frame delta')
    plt.ylabel('Stddev (fraction of PSF core)')
    #plt.ylim(0.7e-3, 3e-3)
    plt.show()
    
    pdb.set_trace()

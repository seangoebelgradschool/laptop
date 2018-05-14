#/usr/bin/env python

#Computes a power spectrum of speckle lifetimes.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
#from scipy.stats import mode
import os

dir = 'images/'
filename = 'saphira_14:46:28.625120914_cleanedaligned.fits'

def interpolate():
    img = pyfits.getdata(dir+filename)
    print "Image read. Size is ", np.shape(img)

    #make sure the first and last frame aren't blank
    i=0
    while np.max(img[i]) == 0:
        i+= 1
    if i>0:
        img=img[i:]
    i=0
    while np.max(img[np.shape(img)[0]-i-1]) == 0:
        i+= 1
    if i>0:
        img=img[:np.shape(img)[0]-i-1]

    
    for i in range(np.shape(img)[0]):
        if i%round(np.shape(img)[0]/10)==0:
            print int(round(np.float(i) / np.shape(img)[0] * 100.)), "% done"
        
        if np.max(img[i]) == 0: #if it's a blank frame
            #search for nearest available legit frames
            no_blanks = 1
            while np.max(img[i+no_blanks])==0:
                no_blanks += 1
            for j in range(no_blanks):
                img[i+j] = img[i-1]*(no_blanks-j)/(no_blanks+1) + \
                           img[i+no_blanks]*(j+1)/(no_blanks+1)

                
    newfilename = dir+filename[:filename.find('.fits')] + 'interped.fits'
    pyfits.writeto(newfilename, img, clobber='true') #save file
    print "Saved as ", newfilename
    
    os.system('ds9 '+ newfilename + ' &')



def powspec():
    

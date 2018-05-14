#/usr/bin/env python
#updated 7/8/15 NEWEST VERSION IS ON SCEXAO

#Decubes a user-specified image and then displays it for easy viewing

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
import os

args=sys.argv
if len(args) != 2: 
    print
    print "Useage: python display.py myfilename.fits"
    print "Displays a cds frame of the image."
    print
    return
else:
    filename=str(args[1])

dir = '../logs/'
img = pyfits.getdata(dir+filename)[0:200, :,:]

img = img + 2**16*(img<0)

done = np.zeros(np.shape(img))
good_images=[] #Blank frames are later deleted.

#figure out where first reset occurs
flux = 0
for i in range(np.shape(img)[0]):
    #print np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]):])
    if np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ]) < flux:
        break
    else:
        flux = np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ])
    first_reset = i+1
    #python says reset = 25 when frame clears at 27 in DS9

#figure out reset frequency
flux = 0
for i in range(first_reset, np.shape(img)[0]):
    if np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ]) < flux:
        break
    else:
        flux = np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ])
reset_freq = i - first_reset

for i in range(200):
    if i%10==0: print "Frame number", i
    if ((i-first_reset)%reset_freq != 0) & ((i-first_reset)%reset_freq != reset_freq-1):
        done[i, :, :] = img[i+1, :, col:col+32] - img[i, :, col:col+32]
        good_images = np.append(good_images, i) #this frame number has data

done = done[good_images.astype(int),:,:] #remove blank frames

newfilename = filename[:filename.find('.fits')] + '_cleaned.fits'
pyfits.writeto(newfilename, done, clobber='true') #save file
print "Saved as ", newfilename

os.system('ds9 '+ newfilename + ' &')

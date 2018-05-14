#/usr/bin/env python
#updated 3/31/15

#Reads in a data ramp .fits file. Subtracts adjacent frames to form
# a cube of cds frames.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

dir = 'saphira/'#danisspeckles/'
filename = '150331_162937.fits'#testarific.fits'

img = pyfits.getdata(dir+filename)
cds = np.zeros((np.shape(img)[0]-1, np.shape(img)[1], np.shape(img)[2]))

for i in range(0,np.shape(img)[0]-1):
    if i%10 == 0: print "Frame", i, 'of',np.shape(img)[0],'.'
    #if i == 0: 
    cds[i,:,:] = img[i,:,:] - img[i+1,:,:]
    #else:
    #    cds = np.vstack((cds, img[i,:,:] - img[i+1,:,:]))

newfile = filename[:filename.find('.fits')] + '_cds.fits'
pyfits.writeto(dir+newfile, cds, clobber='true')
print "File written as", dir+newfile

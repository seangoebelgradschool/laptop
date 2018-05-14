#/usr/bin/env python
#updated 3/31/15 10:48 PM

#Displays a user-specified fits image.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import sys

args=sys.argv
if len(args) != 2: 
    print
    print "Useage: python display.py myfilename.fits"
    print "Displays a cds frame of the image."
    print

else:
    filename=str(args[1])

    img = pyfits.getdata(filename)

    if np.shape(img)[0] == 2:
        cds = img[0,:,:] - img[1,:,:]
    else:
        cds = img[1,:,:] - img[np.shape(img)[0]-1 ,:,:]

    min=np.sort(cds.flatten())[np.size(cds)*0.03]
    max=np.sort(cds.flatten())[np.size(cds)*0.97]

    plt.imshow(cds, vmin=min, vmax=max, interpolation='none')
    plt.title(filename)
    plt.colorbar()
    plt.show()

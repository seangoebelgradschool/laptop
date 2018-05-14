#/usr/bin/env python

import pyfits
import matplotlib.pyplot as plt
import numpy as np
#import pdb
#import os

filename = 'saphira_14:01:45.789966848_cleaned.fits'
dir = 'decubed/'

img = pyfits.getdata(filename)
for i in range(270, np.shape(img)[0]):
    print "saving file", i, 'of', np.shape(img)[0]-1
    plt.imshow(img[i,:,:], interpolation='none', vmin=-150, vmax=950)
    plt.axis('off')
    plt.savefig(dir+'decubed'+str(i)+'.png') #save file

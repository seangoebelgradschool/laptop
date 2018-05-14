#/usr/bin/env python

#plots data ramps of multiple gains to see what range the ADUs cover

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

dir = 'saphira/'
files = ['150329_125752.fits', '150328_220034.fits', 
         '150329_111513.fits', '150329_114017.fits']
volts = ['1V common, Gain=1?'        , '-3V common, Gain=2.5?',
         '-6V common, Gain=5.7?'     , '-8V common, Gain=9.9?']

for j in range(len(files)):
    file = files[j]
    print "file:", file
    im = pyfits.getdata(dir+file)
    flux=[]
    for i in range(3, np.shape(im)[0]):
        flux = np.append(flux, np.median(im[2,:,:] - im[i,:,:] ))
    print j, volts[j]    
    plt.plot(flux, '.')#, label=volts[j])
plt.legend(volts, loc='upper left')
plt.xlabel('CDS Delta')
plt.ylabel('Median ADU')
plt.show()

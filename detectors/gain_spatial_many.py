#/usr/bin/env python

#Calculates gain by looking at the spatial variance and median flux of crops
# from illuminated data sets.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

#H4RG
dir='/home/H4RG/Data/20140102124546/'
filename_sub = 'H4RG_R01_M01_N'

#H2RG
#dir = 'ishell_h2rg/'
#filename_sub = 'ega_gain-00022.Z.'

#crosstalk = 0.011 #for the crosstalk correction
#files = np.arange(10, 40, 1) #frame numbers to loop over

#print "Crosstalk correction factor is", (str(crosstalk*100))[0:4], '%.'
#correction = (5.*crosstalk + 1.) / (crosstalk + 1.)

f40 = pyfits.getdata(dir+filename_sub+'50.fits')
f9 = pyfits.getdata(dir+filename_sub+'30.fits')
cds = f9-f40

#Define reference pixels
ref = [0,1,2,3, len(cds)-1, len(cds)-2, len(cds)-3, len(cds)-4]
#subtract reference pixels at top and bottom
for j in range((cds.shape)[1]):
    cds[ : , j] -= np.mean(cds[ref, j])

var_cube = np.zeros(((cds.shape[0]-8)/5, (cds.shape[1]-8)/5))
median_cube = np.zeros(((cds.shape[0]-8)/5, (cds.shape[1]-8)/5))

#print (cds.shape[0]-8)/5

for i in range((cds.shape[0]-8)/5):
    if (i % round((cds.shape[0]-8.)/5./100.*3.)) == 0:
        print int(round(float(i) / ((cds.shape[0]-8.)/5./100.))), "% done"
    #print i
    for j in range((cds.shape[1]-8)/5):
        crop = cds[i*5+4 : i*5+4 +5 , j*5+4 : j*5+4 +5]
        #remove four elements
        #ok = np.where( (crop > (np.sort(crop.flatten()))[ 0]) & 
        #               (crop < (np.sort(crop.flatten()))[24]) )

        #Select central 60% of pixels, which hopefully don't include outlier populations
        mask = np.where( (crop >  (np.sort(crop.flatten()))[int(round(crop.size * 0.2))]) & 
                         (crop <  (np.sort(crop.flatten()))[int(round(crop.size * 0.8))]) )

        #calculate mask with bad pixels in place
        mask = np.where( (crop > np.median(crop[mask]) - 5.*np.std(crop[mask], ddof=1)) &
                         (crop < np.median(crop[mask]) + 5.*np.std(crop[mask], ddof=1)) )

        var_cube[i,j]    = np.var(crop[mask], ddof=1)
        median_cube[i,j] = np.median(crop[mask])

gain = median_cube/var_cube

print "gain:", np.median(gain)

plt.hist(gain.flatten(), bins=100 , range=[-1,5])
plt.show()

pdb.set_trace()

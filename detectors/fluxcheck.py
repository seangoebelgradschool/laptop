#/usr/bin/env python

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

dir = 'h4rg/20160601/'
filename = 'H4RG_R01_M01_N'


im9 = pyfits.getdata(dir+filename+'09.fits')
center = len(im9)/2

frame_no = np.arange(10, 40, 1)
flux = np.zeros(len(frame_no))

for i in range(len(frame_no)):
    if i%10 == 0: print "frame:", frame_no[i]
    img = pyfits.getdata(dir+filename+str(frame_no[i])+'.fits')
    crop = -1*(im9 - img)[center : center+64 , center : center+64]
    #maskcrop = mask_img[20:90 , 1089:1089+63]

    flux[i] = np.median(crop)#[np.where(maskcrop == 1)])
    #var[i] = np.var(crop[np.where(maskcrop == 1)], ddof=1)

    #plt.figure(num=1, figsize=(4, 8), dpi=100)
    #plt.subplot(211)
    #plt.imshow(crop, interpolation='nearest')
    #plt.colorbar()

    #plt.subplot(212)
    #plt.imshow(maskcrop, vmin=0, vmax=1, interpolation='nearest')
    #plt.colorbar()
    #plt.title('Mask. Red = good, blue = bad')
    #plt.show()

#plt.figure(num=1, figsize=(8, 6), dpi=100) #make it big

#flux plot
#plt.subplot(211)
plt.plot(frame_no-min(frame_no)+1, flux, 'o')
plt.xlabel('CDS delta')
plt.ylabel('Flux [ADU]')

#variance plot
#plt.subplot(212)
#plt.plot(frame_no-np.min(frame_no)+1, var, 'o')
#plt.xlabel('CDS delta')
#plt.ylabel('Variance [ADU^2]')

plt.show()
pdb.set_trace()

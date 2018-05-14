#/usr/bin/env python
#Updated 4/29/14 11:20 PM, started to update on 2/9/15, then realized that
#THIS PROGRAM IS BORKED. METHOD IS BAD. GIVES BAD ANSWER. BAAAAD.

#Calculates the dark current of images

import pyfits
import matplotlib.pyplot as plt
import numpy as np

dir = 'saphira/'
filename = '150206_120328.fits'

img = pyfits.getdata(dir+filename)
flux = []
files = np.arange(10, 200, 1)

ref = [0,1,2,3, len(f9)-1, len(f9)-2, len(f9)-3, len(f9)-4]


for i in files:
    #update progress
    if i % 10 == 0:
        print str(int(round(float(i) / len(files) * 100.)))+ '% done.'

    cds = img[9,:,:] - img[i, :, :]
    
    #Subtract reference pixels
    #for i in range(len(cds)):
    #    cds[: , i] -= np.mean(cds[ref, i])
    flux.append(np.median((cds)[1000:1500 , 1000:1500]))

    time = (files-9.) / 32. * 5.e-6 * (len(cds))**2

plt.plot(time, flux, 'o')
plt.rcParams.update({'font.size':14})
plt.title('Dark Current ('+filename_sub+r1+')')
plt.xlabel('Time (s)')
plt.ylabel('Median Flux (ADU)')

coeffs = np.polyfit(time, flux, 1)
fit = np.poly1d(coeffs)
ys = fit(time)

plt.plot(time, ys)
plt.text(0.1*np.max(time), 0.9*np.max(flux), 
         'y='+str(coeffs[0])[0:4]+'x+'+str(coeffs[1])[0:4])
plt.show()

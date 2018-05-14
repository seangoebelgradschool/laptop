#/usr/bin/env python

#Computes the dark current of a ramp. Averages together first 128 and
# second 128 frames (or the number of frames defined by the n_frames
# variable), then subtracts them to form a CDS frame. Returns
# median dark current in ADU/s for each column. A histogram of pixel
# brightnesses can be plotted by de-commenting the relevant code.
# The mean of the top 4 and bottom 4 rows of reference pixels is
# subtracted.
#
#Script can be called from a unix terminal with the command 
# "python darkcalc.py"

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

dir = 'saphira/'
filename = '150210_155934.fits'
img = pyfits.getdata(dir+filename)

t_exp = 0.5 #seconds per image
n_frames = 200 #frames in each ramp, not including dropped frames. Should be integer.

cds = (np.sum(img[0:n_frames/2 , :, :], 0) -  
       np.sum(img[n_frames/2 : n_frames , :, :], 0)) / (n_frames/2)
cds /= (t_exp * n_frames/2) #convert to ADU/s

print "Median dark current: ", np.median(cds), "ADU/s"

#plot a histogram!
junk = plt.hist(cds.flatten(), bins=200, range=[0,2e3], log='T')
plt.xlabel('ADU/sec of Dark Current')
plt.ylabel('# Occurrences')
plt.show()
pdb.set_trace()





#OLD CODE FOR H4RG ETC

#add together first 128 frames
#ramp = 'M01_N'
#for i in np.arange(9, 9+n_frames, 1):
#    if (i-9) % 5 == 0:
#        print str(int(round((i-9.)/(2.*n_frames)*100.))) + '% done.' #

#    if i == 9:
#        img1 = pyfits.getdata(dir+filename_sub+ramp+'0'+str(i)+'.fits')
#    else:
#        img1 += pyfits.getdata(dir+filename_sub+ramp+str(i)+'.fits')

#add together second 128 frames
#ramp = 'M02_N'
#for i in np.arange(1, 1+n_frames, 1):
#    if (i-1) % 5 == 0:
#        print str(int(round((i-1.+n_frames)/(2.*n_frames)*100.))) + '% done.' 

#    num = str(i)
#    if len(num) == 1:
#        num = '0'+num

#    if i == 1:
#        img2 = pyfits.getdata(dir+filename_sub+ramp+num+'.fits')
#    else:
#        img2 += pyfits.getdata(dir+filename_sub+ramp+num+'.fits')

#n_frames = float(n_frames)
#img1 /= n_frames #compute average frames
#img2 /= n_frames

#create CDS frame
#cds = (img2 - img1) / (t_exp * n_frames) #convert to ADU/second

#define reference pixels
#height = (cds.shape)[0]
#ref = [0,1,2,3, height-1, height-2, height-3, height-4]
#subtract reference pixels
#for j in range((cds.shape)[1]):
#    cds[ : , j] -= np.mean(cds[ref, j])

#for i in np.arange(0, len(cds), 128): #columns
#        print "  Column", i/128+1, np.median(cds[ : , i:i+128])

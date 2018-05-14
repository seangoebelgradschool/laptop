#/usr/bin/env python

#experiment to decube an image with dropped frames

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.stats import mode
#import os

dir = 'images/'
filename = '150729_233701.fits'

print "Reading image..."
img = pyfits.getdata(dir+filename)[:]
print "Image read. Size is ", np.shape(img)

done = np.zeros(np.shape(img))
good_images=np.array([]) #Blank frames are later deleted.


#Figure out reset frequency
#cds = img[:len(img)-1] - img[1:]
meds = np.zeros(np.shape(img)[0])
reset_guesses = np.array([])

for i in range(len(meds)-1): #populate array with medians of each cds frame
    meds[i] = np.median(np.sort((img[i] - img[i+1]).flatten())[0.99*img[i].size: ])
meds = abs(meds - np.median(meds)) #make resets be positive outliers
std = np.std(np.sort(meds)[ : 0.8*meds.size], ddof=1)

for i in range(5,len(meds)): #see where resets occur
    if (meds[i]   > 10.*std) & (meds[i-1] < 10.*std):
        reset_guesses = np.append(reset_guesses, i)
for i in np.arange(30, 2, -1): #check possible reset frequencies
    if mode(reset_guesses[:20] % i)[1][0] > 0.9*len(reset_guesses[:20]):
        reset_freq = i
        first_reset = mode(reset_guesses % i)[0][0]
        break
    if i==3:
        print "Failure."
        pdb.set_trace()
reset_guesses += 1 #dunno, but it's necessary
if first_reset == reset_freq : 
    first_reset = 0 #standard for ./expose
    reset_guesses = np.append(0, reset_guesses)

#plt.plot(meds, 'go')
#plt.plot(reset_guesses, meds[reset_guesses.astype(int)], 'ro')
#plt.show()

#now check if frames have been dropped. Create array of actual reset points
i = 0
resets_actual = np.array([])
while i <= np.shape(img)[0]: #kludgy for loop with varying increments
    if i in reset_guesses: #if the reset was detected previously
        resets_actual = np.append(resets_actual, i)
    else:
        if i+reset_freq in reset_guesses: #just wasn't detected previously, but freq hasn't changed
            resets_actual = np.append(resets_actual, i)
        else: #frame has been dropped
            if i-1 in reset_guesses:
                i -= 1 #decrement by one
                resets_actual = np.append(resets_actual, i)
            else: #shoudn't happen
                print "Code shouldn't have reached here."
                pdb.set_trace()
    i += reset_freq
pdb.set_trace()

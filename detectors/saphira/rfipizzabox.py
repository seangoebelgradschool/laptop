#/usr/bin/env python
#updated 12/9/15

#Decubes an image and then computes the power spectrum of it.
#Differs from rfipowspec because it runs on multiple fits images instead
# of cubes. Also averages frames for subtraction instead of CDSing them.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.stats import mode
import os

def decube():
    dir = 'pizzabox/'
    filename_sub = 'julio-'
    
    readout_rate = 1330. #khz
    
    #create average image
    avgimg = np.zeros(np.shape(pyfits.getdata(dir+filename_sub + '3.fits')))
    frame_nos = np.arange(400,1000) #frame numbers to loop over
    for i in frame_nos:
        avgimg += pyfits.getdata(dir + filename_sub + str(i) + '.fits')
    avgimg /= len(frame_nos)

    if 0:
        resets = get_resets_2(img)
    else: #never reset
        resets = np.array([0]) #reset only at beginning 
    #print "First_reset =", first_reset, ", Reset_freq =", reset_freq

    n_avg = 0 #number of frames included in averaged power spectrum
    for i in frame_nos:
        if (i not in resets) & (i+1 not in resets):
            n_avg += 1
            if 0: #simulate noise to see how it propagates
                #frame = np.random.poisson(lam=100, size=(104, 32))
                #frame = np.random.normal(loc=100, scale=10, size=(104, 32))
                frame = np.random.uniform(low=0, high=10, size=(104, 32))
            else: #use actual data
                frame = avgimg - pyfits.getdata(dir+filename_sub+str(i)+'.fits') #- \
                        #pyfits.getdata(dir+filename_sub+str(i+1)+'.fits')
                if i%100==0:
                    mymax = np.sort(frame.flatten())[np.size(frame)*.96]
                    mymin = np.sort(frame.flatten())[np.size(frame)*.01]
                    plt.imshow(frame, interpolation='none', vmin=mymin, vmax=mymax)
                    plt.colorbar()
                    plt.title(str(i))
                    plt.show()

            if 1: #all channels good
                avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                                np.median(frame[:, 1*32:2*32], 1) , 
                                np.median(frame[:, 2*32:3*32], 1) ,
                                np.median(frame[:, 3*32:4*32], 1) , 
                                np.median(frame[:, 4*32:5*32], 1) ,
                                np.median(frame[:, 5*32:6*32], 1) , 
                                np.median(frame[:, 6*32:7*32], 1) , 
                                np.median(frame[:, 7*32:8*32], 1) , 
                                np.median(frame[:, 8*32:9*32], 1) , 
                                np.median(frame[:, 9*32:10*32], 1) ], 'f')
            else: #only look at some channels
                filtered = np.array([0,4,6])
                avg = np.ravel([np.median(frame[:, 0*32+filtered], 1) , 
                                np.median(frame[:, 1*32+filtered], 1) , 
                                np.median(frame[:, 2*32+filtered], 1) ,
                                np.median(frame[:, 3*32+filtered], 1) , 
                                np.median(frame[:, 4*32+filtered], 1) ,
                                np.median(frame[:, 5*32+filtered], 1) , 
                                np.median(frame[:, 6*32+filtered], 1) , 
                                np.median(frame[:, 7*32+filtered], 1) , 
                                np.median(frame[:, 8*32+filtered], 1) , 
                                np.median(frame[:, 9*32+filtered], 1) ], 'f')

            avg -= np.median(avg)

            #avg = []
            #for y in range(np.shape(frame)[0]):
            #    for x in range(np.shape(frame)[1]/32):
            #        #avg = np.append(avg, np.median(frame[y, (2-x)*32:((2-x)+1)*32]))
            #        avg = np.append(avg, np.median(frame[y, x*32:x*32+8]))

            fft = np.fft.fft(avg)
            try:
                powspec
            except NameError:
                powspec = (abs(fft))**2
            else: 
                powspec += (abs(fft))**2

            #plt.plot((abs(fft))**2)
            #plt.title(str(i))
            #plt.show()

    powspec /= n_avg #normalize to number of samples included

    xaxis = np.fft.fftfreq(avg.size, d=1./readout_rate)
     
    plt.figure(num=1, figsize=(17, 5), dpi=100)
    plt.plot(xaxis, powspec)#, 'bo')
    plt.yscale('log')
    plt.title('Power Spectrum '+dir + ' ' +str(np.shape(frame)))
    plt.ylabel('Power (arb. units)')
    plt.xlabel('Frequency (kHz)')
    #plt.ylim((0, 1.4e8))
    plt.show()

    #pdb.set_trace()

    #Save image?
    if 0: 
        done = done[good_images.astype(int),:,:] #remove blank frames

        newfilename = 'images/'+filename[:filename.find('.fits')] + '_cleaned.fits'
        pyfits.writeto(newfilename, done, clobber='true') #save file
        print "Saved as ", newfilename

        os.system('ds9 '+ newfilename + ' &')




def get_resets_2(img):
    print "Calculating resets."
    #adapted from reset_freq_test.py

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
    print "Done."
    return resets_actual




def get_resets_1():
    #old version, replaced by get_resets_2
    #Figure out reset frequency
    if 1: #image is reset during the cube
        meds = np.zeros(200)
        resets = np.array([])
        for i in range(len(meds)): #populate array with medians of each cds frame
            meds[i] = np.median(np.sort((img[i] - img[i+1]).flatten())[0.99*img[i].size: ])
        meds = abs(meds - np.median(meds)) #make resets be positive outliers
        std = np.std(np.sort(meds)[ : 0.8*meds.size], ddof=1)
        for i in range(5,len(meds)): #see where resets occur
            if (meds[i]   > 10.*std) & (meds[i-1] < 10.*std):
                resets = np.append(resets, i)
        for i in np.arange(30, 2, -1): #check possible reset frequencies
            if mode(resets % i)[1][0] > 0.9*len(resets):
                reset_freq = i
                first_reset = mode(resets % i)[0][0]
                break
            if i==3:
                print "Failure."
        if first_reset == reset_freq-1: 
            first_reset = 0 #standard for ./expose

    else: #Image has no resets during the cube
        first_reset=0
        reset_freq = 10000
    print "First_reset =", first_reset, ", Reset_freq =", reset_freq


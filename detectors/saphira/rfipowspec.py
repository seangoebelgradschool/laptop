#/usr/bin/env python
#updated 10/30/15

#Decubes an image and then computes the power spectrum of it

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.stats import mode
import os
#import time

def decube():
    dir = 'images/'#/home/scexao/APDdata/M04055-39/'
    filename = '150729_230116.fits'
    print "Reading image..."
    img = pyfits.getdata(dir+filename)[:1000]
    print "Image read. Size is ", np.shape(img)
    #cube = np.zeros(np.shape(img))

    done = np.zeros(np.shape(img))
    good_images=np.array([]) #Blank frames are later deleted.

    if 1:
        resets = get_resets_2(img)
    else: #never reset
        reset_freq = 10000
    #print "First_reset =", first_reset, ", Reset_freq =", reset_freq

    for i in np.arange(0, np.shape(img)[0]-1): #1000):
        if (i not in resets) & (i+1 not in resets):
            if 0: #simulate noise to see how it propagates
                #frame = np.random.poisson(lam=100, size=(104, 32))
                #frame = np.random.normal(loc=100, scale=10, size=(104, 32))
                frame = np.random.uniform(low=0, high=10, size=(104, 32))
            else: #use actual data
                #frame = img[i+1, :, col:col+32] - img[i, :, col:col+32]
                frame = img[i, :, :] - img[i+1, :, :]
                if 0:
                    mymax = np.sort(frame.flatten())[np.size(frame)*.99]
                    mymin = np.sort(frame.flatten())[np.size(frame)*.01]
                    plt.imshow(frame, interpolation='none', vmin=mymin, vmax=mymax)
                    plt.title(str(i))
                    plt.show()

            #This needs to be hardcoded for efficiency. :-(
            avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                            np.median(frame[:, 1*32:2*32], 1) , 
                            np.median(frame[:, 2*32:3*32], 1) ,
                            np.median(frame[:, 3*32:4*32], 1) , 
                            np.median(frame[:, 4*32:5*32], 1) ], 'f')
            if len(avg) != np.size(frame)/32:
                print "INCORRECT NUMBER OF COLUMNS. IMAGE SIZE HAS CHANGED, BUT THE CODE HASN'T."
                pdb.set_trace()
            #try reversing to see if anything changes
            #avg = np.ravel([np.median(frame[:, 4*32:5*32], 1) , 
            #                np.median(frame[:, 3*32:4*32], 1) , 
            #                np.median(frame[:, 2*32:3*32], 1) ,
            #                np.median(frame[:, 1*32:2*32], 1) , 
            #                np.median(frame[:, 0*32:1*32], 1) ], 'f') 
                      #      np.median(frame[:, 5*32:6*32], 1) , 
                      #      np.median(frame[:, 6*32:7*32], 1) , 
                      #      np.median(frame[:, 7*32:8*32], 1) , 
                      #      np.median(frame[:, 8*32:9*32], 1) , 
                      #      np.median(frame[:, 9*32:10*32], 1) ], 'f')
            avg -= np.median(avg)

            fft = np.fft.fft(avg)
            powspec = (abs(fft))**2
            
            #build average power spectrum
            try:
                powspec_avg
            except NameError:
                powspec_avg = powspec
            else: 
                powspec_avg += powspec

            diff_img = np.zeros(np.shape(frame))
            #xaxis = np.linspace(-265./2., 265./2., len(powspec))
            xaxis = np.fft.fftfreq(avg.size, d=1./265.e3) /1e3 #don't need to rearrange powspec
            if 1: #remove RFI
                
                #I am deeply sorry, please forgive me
                bads =  [150, 151, 152, 153, 154, 224, 225, 226, 254, 255, 256,
                         326, 327, 328, 329, 330]

                if 0:#show frequency channels that will be nulled
                    plt.figure(num=1, figsize=(17, 5), dpi=100)
                    plt.plot(xaxis, powspec, 'o')
                    plt.vlines(xaxis[bads], 0, 1.1*np.max(powspec), colors='r')
                    plt.yscale('log')
                    plt.title('Power Spectrum')
                    plt.xlabel('Channel Number')
                    plt.ylabel('Power [arbitrary units]')
                    plt.show()
                    

                fft[bads]=0

                #inverse fourier transform
                cleaned = np.fft.ifft(fft)#[500:]
                diff = avg - np.real(cleaned)

                #a = time.clock()
                for col in range(np.shape(frame)[1]/32):
                    for y in range(np.shape(frame)[0]):
                        diff_img[y, col*32 : (col+1)*32] = diff[y*np.shape(frame)[1]/32+col]
                #b = time.clock()
                #THIS IS SOMEHOW SLOWER I DON'T UNDERSTAND WHY I FEEL SO BETRAYED
                #diff_img = np.reshape([val for val in diff for blearg in range(32)] ,
                #                      (np.shape(frame)))
                #c = time.clock()
                #print (c-b)/(b-a)
                
                if 0: #before/after comparison
                    plt.figure(num=1, figsize=(15, 4), dpi=100) 
                    plt.subplot(131)
                    mymin = np.sort(frame.flatten())[np.size(frame)*.01]
                    mymax = np.sort(frame.flatten())[np.size(frame)*.99]
                    plt.imshow(frame, interpolation='none', vmin=mymin, vmax=mymax)
                    plt.title("Before")
                    
                    plt.subplot(132)
                    plt.imshow(diff_img, interpolation='none', vmin=mymin, vmax=mymax)
                    plt.title("Difference image")
                    #plt.colorbar()
                     
                    plt.subplot(133)
                    plt.imshow(frame - diff_img, interpolation='none', vmin=mymin, vmax=mymax)
                    plt.title("After")
                    plt.show()
                    

            done[i, :, :] = frame - diff_img
            good_images = np.append(good_images, i) #this frame number has data

    powspec_avg /= len(good_images) #normalize to number of samples included
    bads = 
    #powspec = np.append(powspec[len(powspec)/2 :], powspec[:len(powspec)/2+1])
    plt.figure(num=1, figsize=(17, 5), dpi=100)
    plt.plot(xaxis, powspec_avg, 'bo')
    plt.vlines(xaxis[bads], 0, 1.1*np.max(powspec), colors='r')
    plt.yscale('log')
    plt.title('Uncorrected Power Spectrum '+filename + ' ' +str(np.shape(frame)))
    plt.ylabel('Power (arb. units)')
    plt.xlabel('Frequency (kHz)')
    plt.show()

    #Save image?
    if 0: 
        done = done[good_images.astype(int),:,:] #remove blank frames

        newfilename = 'images/'+filename[:filename.find('.fits')] + '_cleaned.fits'
        pyfits.writeto(newfilename, done, clobber='true') #save file
        print "Saved as ", newfilename

        os.system('ds9 '+ newfilename + ' &')

    pdb.set_trace()
    #select rfi spikes, input them above, null them out


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

    for i in range(4,len(meds)): #see where resets occur
        if (meds[i]   > 10.*std) & (meds[i-1] < 10.*std):
            reset_guesses = np.append(reset_guesses, i)
    reset_guesses += 1 #dunno, but it's necessary
    for i in np.arange(30, 2, -1): #check possible reset frequencies
        if mode(reset_guesses[:20] % i)[1][0] > 0.9*len(reset_guesses[:20]):
            reset_freq = i
            first_reset = mode(reset_guesses % i)[0][0]
            break
        if i==3:
            print "Failure."
            pdb.set_trace()
    if first_reset == reset_freq : 
        first_reset = 0 #standard for ./expose

    i=0
    while i+first_reset < np.min(reset_guesses): #dunno why this happens
        reset_guesses = np.append(i+first_reset, reset_guesses)
        i += reset_freq

    #plt.plot(meds, 'go')
    #plt.plot(reset_guesses, meds[reset_guesses.astype(int)], 'ro')
    #plt.show()

    #now check if frames have been dropped. Create array of actual reset points
    i = 0
    resets_actual = np.array([])
    cont = True
    while (i <= np.shape(img)[0]) & (cont == True): #kludgy for loop with varying increments
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
                    print "Bunch of frames dropped around"+str(i)+". Giving up."
                    cont = False
                    #print "Code shouldn't have reached here."
                    plt.plot(meds[i-50:i+50], 'o')
                    plt.show()
                    pdb.set_trace()
        i += reset_freq
    print "Done."
    return resets_actual

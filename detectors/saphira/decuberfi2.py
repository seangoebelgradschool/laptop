#/usr/bin/env python
#This code was forked on 4/21/16 from rfipowspec. Instead
# of dropping frames around a reset, it replaces them with blank
# ones in order to keep the timing consistent. It also replaces
# dropped frames with blank ones. Unlike decuberfi, this uses
# timestamp information.
#updated 05/28/16

#Performs some combination of decubing an image, computing
# the power spectrum of it, and Fourier Filtering away RFI
#This is a refactored version nof rfipowspec.py.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.stats import mode
import os
#import time

#Declare some global variables
dir = 'images/'
filename = 'saphira_14:46:28.625120914.fits'
#saphira_14:46:21.006342721.fits'
#saphira_14:46:51.541667336.fits'
print "Reading image..."
img = pyfits.getdata(dir+filename)#[5000:]
print "Image read. Size is ", np.shape(img)
readout_rate = 265. #kpix/s/channel, standard for leach controller
#global img, dir, filename, readout_rate


#####################################################################

#The following three procedures are what you want to call. The others
# hopefully don't need to be derped with.

#####################################################################

def decube_save():
#decubes an image and saves it

    cube = decube()
    save(cube)

def decube_powspec():
#decubes an image. Computes power spectrum of it. Identifies and saves
# frequencies that RFI occurs. Must be run before decube_remove_rfi().

    cube = decube()
    
    compute_fft(cube, cleanrfi=False)
    #save(cube)

def decube_remove_rfi():
#decubes an image. Restores frequencies of RFI as identified by
# decube_powspec(). Nulls them away using Fourier Filtering. Requires
# decube_powspec() to be run first.

    cube = decube()

    #150729_233701.fits
    #bads = [159, 160, 161, 205, 206, 224, 225, 255, 256, 274, 275,
    #        319, 320, 321]
    
    #150729_230116.fits
    bads = [31, 149, 150, 151, 152, 153, 154, 223, 224, 225, 226,
            254, 255, 256, 257, 326, 327, 328, 329, 330, 331, 449]
    
    #I am deeply sorry, please forgive me
    #150729_230116_cleaned.fits
    #bads = [150, 151, 152, 153, 224, 225, 255, 256, 327, 328, 329, 330]

    #saphira_14:00:03.259552239.fits
    #bads = [139,140, 141,142, 170, 171, 172,173]

    #saphira_14:10:04.159651175.fits'
    #bads = [95, 139, 140, 141, 142, 170, 171, 172, 173, 217]
    
    cube = compute_fft(cube, bads=bads, cleanrfi=True)
    save(cube)


#####################################################################


def decube():
    global img#, dir, filename, readout_rate
    
    if 'saphira' in filename: #weaver used
        img = img + 2**16*(img<0) #correct for overflow
        flux_direction = 1 #ADUs increase with time
    else:
        flux_direction = -1
    
    done = np.zeros((np.shape(img)[0]+100, np.shape(img)[1], np.shape(img)[2]))
    #good_images=np.array([]) #Blank frames are later deleted.

    if 1:
        #resets = get_resets_2(img)
        junk = get_resets_2(img)

        goods = junk[0] #which frames are good for CDSing?
        incs = junk[1] #what was the frame interval? 1=normal, 2=1 dropped frame, etc
    else: #never reset
        resets = 3*np.arange(len(img)/3) #3 frame reset, 0th frame is a reset

    dropcount = 0
    for i in range(np.shape(img)[0]-1):
        if (goods[i]==1):
            done[i+dropcount] = flux_direction*(img[i+1] - img[i])
        if incs[i] > 1: dropcount += (incs[i]-1)
    return done



def compute_fft(image, bads=1e5, cleanrfi=False):
    if (bads == 1e5) & (cleanrfi == True):
        print "No RFI frequencies given to null, but you want me to do this anyway?"
        pdb.set_trace()
        
    if len(np.shape(image)) == 2: #if 2d image
        frame = np.reshape(image, np.append(1, np.shape(image))) #make it a single-layer 3D image
    
    done=np.zeros(np.shape(image)) #only needed if removing RFI...

    for i in range(np.shape(image)[0]):
        if i % (np.shape(image)[0] /10) == 0: #update progress
            print str(int(round(float(i) / np.shape(image)[0] * 100.)))+ "% complete."
            
        frame = image[i]

        avg = frametoarray(frame)
            
        fft = np.fft.fft(avg)
        xaxis = np.fft.fftfreq(avg.size, d=1./readout_rate) #don't need to rearrange powspec
        #^should this be moved elsewhere?

        if cleanrfi==False: #If we should calculate an average power spectrum
            powspec = (abs(fft))**2

            #build average power spectrum
            try:
                powspec_avg
            except NameError:
                powspec_avg = powspec
            else: 
                powspec_avg += powspec
        else: #if we should null frequencies
            diff_img = np.zeros(np.shape(frame))

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
            #THIS IS SOMEHOW SLOWER. I DON'T UNDERSTAND WHY. I FEEL SO BETRAYED.
            #diff_img = np.reshape([val for val in diff for blearg in range(32)] ,
            #                      (np.shape(frame)))
            #c = time.clock()
            #print (c-b)/(b-a)

            done[i, :, :] = frame - diff_img

            if 0: #before/after comparison
                plt.figure(num=1, figsize=(15, 4), dpi=100) 
                plt.subplot(131)
                mymin = np.sort(frame.flatten())[np.size(frame)*.01]
                mymax = np.sort(frame.flatten())[np.size(frame)*.9]
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

    if cleanrfi==False: #If we're computing an average power spectrum
        powspec_avg /= np.shape(image)[0] #normalize by number of samples included

        if bads == 1e5: #if we should calculate the RFI frequencies
            flatsort = np.sort(frame.flatten())

            if flatsort[len(flatsort)-20] > flatsort[0.35*len(flatsort)] + \
               10 * np.std(flatsort[: 0.7*len(flatsort)]):
                psf=True
                print "PSF detected."
            else:
                psf=False
                print "No PSF detected."
            bads = findrfi(powspec_avg, xaxis, psf)

        #powspec = np.append(powspec[len(powspec)/2 :], powspec[:len(powspec)/2+1])
        
        #print bads
        plt.figure(num=1, figsize=(17, 5), dpi=100)
        plt.plot(xaxis, powspec_avg, 'bo')
        plt.vlines(xaxis[bads], 0, 1.1*np.max(powspec), colors='r')
        plt.yscale('log')
        plt.title('Uncorrected Power Spectrum '+filename + ' ' +str(np.shape(frame)))
        plt.ylabel('Power (arb. units)')
        plt.xlabel('Frequency (kHz)')
        plt.show()
        
    else: #if RFI nulled
        return done


def save(cube):
    newfilename = dir+filename[:filename.find('.fits')] + '_cleaned.fits'
    pyfits.writeto(newfilename, cube, clobber='true') #save file
    print "Saved as ", newfilename
    
    os.system('ds9 '+ newfilename + ' &')


def findrfi(powspec, xaxis, psf):
    #ID spikes in powspec. 1e5 is arbitrary. Don't bother with <10 khz
    bads = np.array(np.where((powspec > 3e5) & (abs(xaxis) > 14))).flatten()

    if psf==True:
        n_cols = np.shape(img)[2] / 32.
        print "N_columns:", n_cols

        n=1.
        nonlegit = np.array([]) #indices of spikes in powspec caused by psf
        while readout_rate/n_cols*n < readout_rate/2. : #within range of power spectrum
            if readout_rate/n_cols*n in xaxis[bads]: #if PSF causes spike in power spectrum
                loc = np.where(xaxis[bads] == readout_rate/n_cols*n)
                #nonlegit = np.append(nonlegit, bads[loc])
                i=0
                while bads[loc]+i in bads:
                    nonlegit = np.append(nonlegit, np.array(loc)+i)
                    i+=1
                i = -1
                while bads[loc]+i in bads:
                    nonlegit = np.append(nonlegit, np.array(loc)+i)
                    i-=1
                    
                n *= -1. #now do the negative side of the power spectrum
                loc = np.where(xaxis[bads] == readout_rate/n_cols*n)
                #nonlegit = np.append(nonlegit, bads[loc])
                i=0
                while bads[loc]+i in bads:
                    nonlegit = np.append(nonlegit, np.array(loc)+i)
                    i+=1
                i = -1
                while bads[loc]+i in bads:
                    nonlegit = np.append(nonlegit, np.array(loc)+i)
                    i-=1
                n *= -1. #return to being positive
            n+=1. #fourier aliasing
        bads = np.delete(bads, nonlegit) #remove non-legit elements from bads
        print "bads:", bads
        
        #find indices of local maxima of identified
        #localmax = bads [np.r_[True, powspec[bads][1:] > powspec[bads][:-1]] & \
        #                 np.r_[powspec[bads][:-1] > powspec[bads][1:], True] ]

    return bads



def get_resets_2(img):
    print "Calculating resets."
    #adapted from reset_freq_test.py

    #Figure out reset frequency
    #cds = img[:len(img)-1] - img[1:]
    meds = np.zeros(np.shape(img)[0])
    reset_guesses = np.array([])

    for i in range(len(meds)-1): #populate array with medians of each cds frame
        meds[i] = np.sort((img[i] - img[i+1]).flatten())[0.995*np.size(img[i])]

    meds = abs(meds - np.median(meds)) #make resets be positive outliers
    meds -= np.sort(meds)[.3*len(meds)] #make most data points around 0
    std = np.std(np.sort(meds)[ : 0.6*len(meds)], ddof=1)

    for i in range(1,len(meds)): #see where resets occur
        if (meds[i] > 10.*std):# & (meds[i-1] < 10.*std):
            reset_guesses = np.append(reset_guesses, i)
    reset_guesses += 1 #dunno, but it's necessary
    for i in np.arange(30, 3, -1): #check possible reset frequencies
        if mode(reset_guesses[:30] % i)[1][0] > 0.4*len(reset_guesses[:30]):
            #mymode = mode(reset_guesses[:30] % i)[0][0]
            #if list(reset_guesses[:30] % i).count(mymode) + \
            #   list(reset_guesses[:30] % i).count(mymode+1) > \
            #   0.9*len(reset_guesses[:30]):
            if mode(reset_guesses[:30] % (i/2.))[1][0] > 0.8*len(reset_guesses[:30]):
                i = i/2
            reset_freq = i
            first_reset = mode(reset_guesses[:30] % i)[0][0]
            break
    else: #you can close a for loop with an else statement!
        print "Failure."
        pdb.set_trace()
    
    if first_reset == reset_freq : 
        first_reset = 0 #standard for ./expose

    print "Reset interval:", reset_freq
        
    i=0
    while i+first_reset < np.min(reset_guesses): #dunno why this happens
        reset_guesses = np.append(i+first_reset, reset_guesses)
        i += reset_freq

    goods = np.zeros(len(meds)) #which frames are good? 1=good, 0=bad
    incs =  np.zeros(len(meds)) #how many frames should there be in this increment?
    #goods[i] indicates whether frame[i] is worth CDSing
    #incs[i] indicates the frame increment between frame[i] and [i+1].
    # Last element is empty.
    #intervals[i] corresponds to the time diff between frames [i] and [i+1]
    intervals = get_timestamps()

    i=0
    while i+reset_freq < len(meds):
        #check simplest case first
        if (np.min(meds[i]) > 15.*std) & \
           (np.max(meds[i+1:i+reset_freq-1]) < 15.*std) & \
           (meds[i+reset_freq] > 15.*std) & \
           (sum(intervals[i:i+reset_freq]) > reset_freq-1) & \
           (sum(intervals[i:i+reset_freq]) < reset_freq+1): 
            #in order, that was
            #first frame after reset is bad
            #next bunch of frames are good
            #another reset detected
            #timing is reasonable
            goods[i] = 0
            goods[i+1 : i+reset_freq-1] = 1
            goods[i+reset_freq-1] = 0
            incs[i : i+reset_freq] = 1
            i += reset_freq 
        else: #go frame by frame
            if meds[i] < 15.*std:
                goods[i] = 1
            incs[i] = round(intervals[i])
            i += 1
    for i in np.arange(len(meds)-reset_freq, len(meds)-1):
        if meds[i] < 15.*std:
            goods[i] = 1
        incs[i] = round(intervals[i])
            
    print "Done finding resets."
    print "Sampling rate is", int(round(readout_rate*1e3/(np.size(img[0])/32))), "Hz", \
        "with a", int(round((reset_freq-2.)/reset_freq*100.)), "% duty cycle."

    return [goods, incs] #returns a (2,len(meds)) size array

def get_timestamps():
    times = np.loadtxt(dir + filename.replace('.fits' , '.txt'), dtype='str')

    nicetimes = np.zeros(len(times))
    timediffs = np.zeros(len(times)-1)

    fixcount = 0
    for i in range(len(times)):
        #check if seconds decimal incremented but seconds didn't
        #seconds place is times[i][6:8]
        if i < len(times)-1: #if you aren't at the end of the list
            if times[i][9:] > times[i+1][9:]:
                #print "before"
                #print "i:", i
                #print times[i-3:i+7]
                
                if times[i][6:8] == '59':
                    print "WELL CRAP, you should have coded overflow."
                    pdb.set_trace()
                    
                j = 1
                while times[i+j][6:8] <= times[i][6:8]:
                    times[i+j] = times[i+j][:6] + \
                                 str(int(times[i+j][6:8])+1) + \
                                 times[i+j][8:]
                    j += 1
                    fixcount += 1
                #print "after"
                #print times[i-3:i+7]
            
        nicetimes[i] = float(times[i][:2])*3600. + \
                       float(times[i][3:5])*60. + \
                       float(times[i][6:])

    print "This code fixed Oli's bad clock", fixcount, "times."
        
    for i in range(len(timediffs)):
        timediffs[i] = nicetimes[i+1] - nicetimes[i]

    print "According to timestamps, the framerate was about", \
        1. / np.median(timediffs), "Hz."

    timediffs /= np.median(timediffs) #converts it from secs to frame numbers
    
    #myhist = plt.hist(timediffs, bins = np.max(timediffs)/0.1,
    #                  log='True', range=[0, 1.1*np.max(timediffs)])
    #plt.axvline(x=np.median(timediffs), color='r')
    #plt.ylim((0.5, 1e4))
    #plt.xlim((0,np.max(timediffs)+1))
    #plt.show()
    
    return(timediffs)



def frametoarray(frame):
    n_cols = np.shape(frame)[1] / 32
    #This needs to be hardcoded for efficiency. :-(

    if n_cols == 1:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) ], 'f')
                        
    elif n_cols == 2:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) ], 'f')
        
    elif n_cols == 3:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) ], 'f')
        
    elif n_cols == 4:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) , 
                        np.median(frame[:, 3*32:4*32], 1) ], 'f')
    elif n_cols == 5:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) ,
                        np.median(frame[:, 3*32:4*32], 1) , 
                        np.median(frame[:, 4*32:5*32], 1) ], 'f')

    elif n_cols == 6:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) ,
                        np.median(frame[:, 3*32:4*32], 1) , 
                        np.median(frame[:, 4*32:5*32], 1) , 
                        np.median(frame[:, 5*32:6*32], 1) ], 'f')

    elif n_cols == 7:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) , 
                        np.median(frame[:, 3*32:4*32], 1) , 
                        np.median(frame[:, 4*32:5*32], 1) , 
                        np.median(frame[:, 5*32:6*32], 1) , 
                        np.median(frame[:, 6*32:7*32], 1) ], 'f')

    elif n_cols == 8:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) , 
                        np.median(frame[:, 3*32:4*32], 1) , 
                        np.median(frame[:, 4*32:5*32], 1) , 
                        np.median(frame[:, 5*32:6*32], 1) , 
                        np.median(frame[:, 6*32:7*32], 1) , 
                        np.median(frame[:, 7*32:8*32], 1) ], 'f')

    elif n_cols == 9:
        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) , 
                        np.median(frame[:, 3*32:4*32], 1) , 
                        np.median(frame[:, 4*32:5*32], 1) , 
                        np.median(frame[:, 5*32:6*32], 1) , 
                        np.median(frame[:, 6*32:7*32], 1) , 
                        np.median(frame[:, 7*32:8*32], 1) , 
                        np.median(frame[:, 8*32:9*32], 1) ], 'f')

    elif n_cols == 10:
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
    else:
        print "Code should never have reached here, something is terribly wrong."
        pdb.set_trace()

    avg -= np.median(avg)

    if len(avg) != np.size(frame)/32:
        print "catastrophic error!"
        pdb.set_trace()
    
    if 0: #power spectrum by eye
        plt.plot(avg)
        plt.show()
    
    return avg

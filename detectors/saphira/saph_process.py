#/usr/bin/env python

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
#import os

#dir = '/media/data/20150402/saphira_data/'
filename = 'saphira_14:01:45.789966848.fits'
img = pyfits.getdata(filename)
#cube = np.zeros(np.shape(img))

#correct for annoying integer overflow issue
global img
print "Fixing overflow issue..."
img = img + 2**16*(img<0)
print "Fixed overflow issue."

def processer():
    reset_freq = 50

    done = np.zeros(np.shape(img))
    good_images=[] #Blank frames are later deleted.

    #correct for annoying integer overflow issue
    #global img
    #print "Fixing overflow issue..."
    #img = img + 2**16*(img<0)
    #print "Fixed overflow issue."

    #figure out where first reset occurs
    flux = 0
    for i in range(np.shape(img)[0]):
        #print np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]):])
        if np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]):]) < flux:
            break
        else:
            flux = np.median(np.sort(img[i,:,:].flatten())[0.93*np.size(img[i,:,:]):])
        first_reset = i+1
        #python says reset = 25 when frame clears at 27 in DS9
    
    for i in range(1000):#np.shape(done)[0]): #n_frames
        print "Frame number", i
        if ((i-first_reset)%reset_freq != 0) & ((i-first_reset)%reset_freq != reset_freq-1):
            for col in range(0, np.shape(img)[2], 32):
                frame = img[i+1, :, col:col+32] - img[i, :, col:col+32]
                done[i, :, col:col+32] = clean(frame)
            good_images = np.append(good_images, i) #this frame number has data

            if 1:
                plt.subplot(121)
                mymin = np.min(done[i,:,:])
                mymax = np.max(done[i,:,:])
                plt.imshow(img[i+1, :, :] - img[i, :, :], interpolation='none', 
                           vmin=mymin, vmax=mymax)
                plt.title(str(i) + ", Before")
                
                plt.subplot(122)
                plt.imshow(done[i,:,:], interpolation='none', vmin=mymin, vmax=mymax)
                plt.title(str(i) + ", After")
                #plt.colorbar()
                plt.show()

    done = done[good_images.astype(int),:,:] #remove blank frames

    newfilename = filename[:filename.find('.fits')] + '_cleaned.fits'
    pyfits.writeto(newfilename, done, clobber='true') #save file

def makepowspec():
    for i in range(2,25):
        for col in range(0, np.shape(img)[2], 64):
            
            if 0: #simulate noise to see how it propagates
                #frame = np.random.poisson(lam=100, size=(104, 32))
                #frame = np.random.normal(loc=100, scale=10, size=(104, 32))
                frame = np.random.uniform(low=0, high=10, size=(104, 32))
            else: #use actual data
                frame = img[i+1, :, col:col+32] - img[i, :, col:col+32]

            avg = np.median(frame, 1)
            avg -= np.median(avg)
        
            arr = avg#np.append(np.zeros(500),avg)
            fft = np.fft.fft(arr)

            #if i == 4 and col == 0:
            powspec = (abs(fft))**2
            #else: 
            #    powspec += (abs(fft))**2

            xaxis = np.arange(len(fft)) - 0.5*len(fft)
            plt.figure(num=1, figsize=(17, 5), dpi=100)            
            plt.plot(xaxis, powspec, 'b-')
            bad = [4,20, 28, 36, 44]
            #bad = [24, 69, 116, 163, 208, 256]
            plt.vlines(bad, 0, np.max(powspec), colors='r')
            #plt.yscale('log')
            #plt.xlim(0,104)
            plt.xlim(xmin=-1)
            plt.title('frame '+str(i)+', col '+str(col))
            plt.show()

            #pdb.set_trace()


def clean(frame):
    #remove bad frequencies

    avg = np.median(frame, 1)
    avg -= np.median(avg)
        
    arr = avg#np.append(np.zeros(500),avg)
    fft = np.fft.fft(arr)

    powspec = (abs(fft))**2

    bads = []
    badguesses = np.array([4, 13, 20, 28, 36, 44]) + 0.5*len(fft)
    for guess in badguesses:
        peak = np.max(np.where(powspec == np.max(powspec[guess-1:guess+2])))
        bads = np.append(bads, peak)
        if powspec[peak-1] > 10*(powspec[peak-2]+powspec[peak-3]):
            bads = np.append(bads, peak-1)
        if powspec[peak+1] > 10*(powspec[peak+2]+powspec[peak-3]):
            bads = np.append(bads, peak+1)

    xaxis = np.arange(len(fft)) - 0.5*len(fft)

    if 1:#show frequency channels that will be nulled
        plt.figure(num=1, figsize=(17, 5), dpi=100)
        plt.plot(xaxis, powspec)
        plt.vlines(bads-len(fft)/2, 0, 1.1*np.max(powspec), colors='r')
        plt.vlines(-1*bads+len(fft)/2, 0, np.max(powspec), colors='r')
        plt.xlim(-1.*len(fft)/2, len(fft)/2)
        plt.title('Power Spectrum')
        plt.xlabel('Channel Number')
        plt.ylabel('Power [arbitrary units]')
        plt.show()

    
    for bad in bads:
        fft[   bad ] = 0
        fft[-1*bad + len(fft)] = 0
        
    powspec = (abs(fft))**2

    #inverse fourier transform
    cleaned = np.fft.ifft(fft)#[500:]
    diff = avg - np.real(cleaned)

    cleaned = np.zeros(np.shape(frame))
    for y in range(np.shape(frame)[0]):
        cleaned[y,:] = frame[y,:] - diff[y]

    if 0:
        #plt.figure(num=1, figsize=(5, 5), dpi=100)
        plt.subplot(131)
        mymin = np.min(frame)
        mymax = np.max(frame)
        plt.imshow(frame, interpolation='none', vmin=mymin, vmax=mymax)
        plt.title("Before")
        
        plt.subplot(132)
        plt.imshow(cleaned, interpolation='none', vmin=mymin, vmax=mymax)
        plt.title("After")
        
        plt.subplot(133)
        plt.imshow(np.transpose(np.tile(diff, (32,1))), 
                   interpolation='none', cmap='gray')
        plt.title("Correction")
        plt.show()
    return cleaned

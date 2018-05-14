#/usr/bin/env python
#updated 7/27/15

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
#import os

#dir = '/home/scexao/APDdata/M04055-39/'
dir = 'cubes/'
filename = '150731_052131_cleaned.fits'
img = pyfits.getdata(dir+filename)#[2000:,:,:]
#cube = np.zeros(np.shape(img))

#correct for annoying integer overflow issue
global img
#print "Fixing overflow issue..."
#img = img + 2**16*(img<0)
#print "Fixed overflow issue."

def processer():
    done = np.zeros(np.shape(img))
    good_images=np.array([]) #Blank frames are later deleted.

    #ENABLE FOR WEAVER
    if 0:
        #correct for annoying integer overflow issue
        #global img
        #print "Fixing overflow issue..."
        #img = img + 2**16*(img<0)
        #print "Fixed overflow issue."
        
        #figure out where first reset occurs
        flux = 0
        for i in range(np.shape(img)[0]):
            #print np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]):])
            if np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ]) < flux:
                break
            else:
                flux = np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ])
            first_reset = i+1
            #python says reset = 25 when frame clears at 27 in DS9

        #figure out reset frequency
        flux = 0
        for i in range(first_reset, np.shape(img)[0]):
            if np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ]) < flux:
                break
            else:
                flux = np.median(np.sort(img[i,:,:].flatten())[0.95*np.size(img[i,:,:]) : ])
    else: #FOR IMAGES TAKEN WITH ./EXPOSE
        #doesn't always reset to the same point?
        first_reset = 0 #beginning of cube
        #figure out reset frequency
        flux = 1e5
        for i in range(first_reset, np.shape(img)[0]):
            if np.median(np.sort(img[i,:,:].flatten())[0.99*np.size(img[i,:,:]) : ]) > flux:
                break
            else:
                flux = np.median(np.sort(img[i,:,:].flatten())[0.99*np.size(img[i,:,:]) : ])
    reset_freq = i - first_reset

    reset_freq = 10
    #print first_reset, reset_freq
    #pdb.set_trace()

    for i in range(1000):#np.shape(done)[0]-1): #n_frames
        if i%10==0: print "Frame number", i
        if ((i-first_reset)%reset_freq != 0) & ((i-first_reset)%reset_freq != reset_freq-1):
            #DECOMMENT THESE LINES TO PROCESS STRIPES INSTEAD OF WHOLE FRAME
            #for col in range(0, np.shape(img)[2], 32):
            #    frame = img[i+1, :, col:col+32] - img[i, :, col:col+32]
            #    done[i, :, col:col+32] = clean(frame)
            #COMMENT THESE LINES TO PROCESS STRIPES INSTEAD OF WHOLE FRAME
            frame = img[i+1, :, :] - img[i, :, :]
            done[i, :, :] = frame#clean(frame)

            good_images = np.append(good_images, i) #this frame number has data

            if 0:
                plt.figure(num=1, figsize=(10, 5), dpi=100) 
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

    newfilename = 'cubes/'+filename[:filename.find('.fits')] + '_cleaned.fits'
    pyfits.writeto(newfilename, done, clobber='true') #save file
    print "Saved as ", newfilename

def makepowspec():
#Old. Replaced by rfipowspec.py.
    for i in range(2,100):
        if 0: #simulate noise to see how it propagates
                #frame = np.random.poisson(lam=100, size=(104, 32))
                #frame = np.random.normal(loc=100, scale=10, size=(104, 32))
            frame = np.random.uniform(low=0, high=10, size=(104, 32))
        else: #use actual data
                #frame = img[i+1, :, col:col+32] - img[i, :, col:col+32]
            frame = img[i, :, :]# - img[i, :, :]
            #frame -= np.median(frame[:,0:8])

        avg = np.ravel([np.median(frame[:, 0*32:1*32], 1) , 
                        np.median(frame[:, 1*32:2*32], 1) , 
                        np.median(frame[:, 2*32:3*32], 1) ,
                        np.median(frame[:, 3*32:4*32], 1) ], 'f')
       #                 np.median(frame[:, 4*32:5*32], 1) , 
       #                 np.median(frame[:, 5*32:6*32], 1) , 
       #                 np.median(frame[:, 6*32:7*32], 1) , 
        #                np.median(frame[:, 7*32:8*32], 1) , 
        #                np.median(frame[:, 8*32:9*32], 1) , 
        #                np.median(frame[:, 9*32:10*32], 1) ], 'f')
        #avg = np.ravel([np.median(frame[:, 0*32:0*32+8], 1) , 
        #                np.median(frame[:, 1*32:1*32+8], 1) , 
        #                np.median(frame[:, 2*32:2*32+8], 1) ,#], 'f')
        #                np.median(frame[:, 3*32:3*32+8], 1) , 
        #                np.median(frame[:, 4*32:4*32+8], 1) , 
        #                np.median(frame[:, 5*32:5*32+8], 1) , 
        #                np.median(frame[:, 6*32:6*32+8], 1) , 
        #                np.median(frame[:, 7*32:7*32+8], 1) , 
        #                np.median(frame[:, 8*32:8*32+8], 1) , 
        #                np.median(frame[:, 9*32:9*32+8], 1) ], 'f')
        avg -= np.median(avg)
        #avg = []
        #for y in range(np.shape(frame)[0]):
        #    for x in range(np.shape(frame)[1]/32):
        #        #avg = np.append(avg, np.median(frame[y, (2-x)*32:((2-x)+1)*32]))
        #        avg = np.append(avg, np.median(frame[y, x*32:x*32+8]))
        #pdb.set_trace()
        arr = avg
        fft = np.fft.fft(arr)
        
        if (i == 2):
            powspec = (abs(fft))**2
        else: 
            powspec += (abs(fft))**2
            
    powspec = np.append(powspec[len(powspec)/2 :], powspec[:len(powspec)/2+1])
    #xaxis = np.arange(len(fft)) - 0.5*len(fft)
    xaxis = np.linspace(-265./2., 265./2., len(powspec))
    plt.figure(num=1, figsize=(17, 5), dpi=100)
    plt.plot(xaxis, powspec, 'b-')
    #bad = [24, 69, 116, 163, 208, 256)
    #plt.vlines(bad, 0, np.max(powspec), colors='r')
    #line = np.array([100])
    #plt.vlines(np.append(line, -1*line), 0, np.max(powspec), colors='r')
    plt.yscale('log')
    #plt.xlim(xmin=-1)
    plt.title('Power Spectrum '+filename)
    plt.ylabel('Power (arb. units)')
    plt.xlabel('Frequency (kHz)')
    plt.show()



def clean(frame):
    #remove bad frequencies

    #avg = np.median(frame, 1)
    avg = np.zeros(np.shape(frame)[0] * np.shape(frame)[1] / 32)
    for y in range(np.shape(frame)[0]):
        for x in range(np.shape(frame)[1]/32):
            avg[x+np.shape(frame)[1]/32*y] = np.median(frame[y , x*32:(x+1)*32])

    avg -= np.median(avg)
    
    arr = avg#np.append(np.zeros(500),avg)
    fft = np.fft.fft(arr)

    powspec = (abs(fft))**2

    #bads = []
    #badguesses = np.array([4, 13, 20, 28, 36, 44]) + 0.5*len(fft)
    #for guess in badguesses:
    #    peak = np.max(np.where(powspec == np.max(powspec[guess-1:guess+2])))
    #    bads = np.append(bads, peak)
    #    if powspec[peak-1] > 10*(powspec[peak-2]+powspec[peak-3]):
    #        bads = np.append(bads, peak-1)
    #    if powspec[peak+1] > 10*(powspec[peak+2]+powspec[peak-3]):
    #        bads = np.append(bads, peak+1)
    bads = np.ravel(np.where(powspec > 1.5*np.std(powspec[3:-2]))) #baseline=0
    #print bads
    if bads[0] == 0: bads = bads[1:]
    if bads[0] == 1: bads = bads[1:]
    if bads[-1] == len(powspec)-1: bads = bads[:-1]
    #print bads

    xaxis = np.arange(len(fft)) - 0.5*len(fft)

    if 0:#show frequency channels that will be nulled
        plt.figure(num=1, figsize=(17, 5), dpi=100)
        plt.plot(xaxis, powspec)
        plt.vlines(xaxis[bads], 0, 1.1*np.max(powspec), colors='r')
        #plt.vlines(-1*bads+len(fft)/2, 0, np.max(powspec), colors='r')
        #plt.xlim(-1.*len(fft)/2, len(fft)/2)
        plt.title('Power Spectrum')
        plt.xlabel('Channel Number')
        plt.ylabel('Power [arbitrary units]')
        plt.show()

    fft[bads]=0
    #for bad in bads:
        #fft[   bad ] = 0
        #fft[-1*bad + len(fft)] = 0
    #powspec = (abs(fft))**2

    #inverse fourier transform
    cleaned = np.fft.ifft(fft)#[500:]
    diff = avg - np.real(cleaned)

    cleaned = np.zeros(np.shape(frame))
    for y in range(np.shape(frame)[0]):
        for x in range(np.shape(frame)[1]/32):
            cleaned[y, x*32:(x+1)*32] = frame[y, x*32:(x+1)*32] - \
                diff[y*np.shape(frame)[1]/32+x]

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

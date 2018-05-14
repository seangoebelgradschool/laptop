#/usr/bin/env python

#Fixes saphira overflow issue. Forms CDS frames, discarding
# those near a reset. Performs Fourier Filtering on the CDS
# frames to reduce RFI noise. 

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

#dir = '/media/data/20150402/saphira_data/'
filename = 'saphira_14:01:45.789966848.fits'
img = pyfits.getdata(filename)

#correct for annoying integer overflow issue
global img
print "Fixing overflow issue..."
img = img + 2**16*(img<0)
print "Fixed overflow issue."

def processer():
    reset_freq = 50

    done = np.zeros(np.shape(img))
    good_images=[] #Blank frames are later deleted.

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
            #for col in range(0, np.shape(img)[2], 32):
            frame = img[i+1, :, :] - img[i, :, :]
            done[i, :, :] = clean(frame)
            good_images = np.append(good_images, i) #this frame number has data
            
            if 0:
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
#        for col in range(0, np.shape(img)[2], 64):
            
        if 0: #simulate noise to see how it propagates
                #frame = np.random.poisson(lam=100, size=(104, 32))
                #frame = np.random.normal(loc=100, scale=10, size=(104, 32))
            frame = np.random.uniform(low=0, high=10, size=(104, 32))
        else: #use actual data
            frame = img[i+1, :, :] - img[i, :, :]

        avg = [np.median(frame[:,   :32], 1), 
               np.median(frame[:, 32:64], 1), 
               np.median(frame[:, 64:  ], 1) ] #a 2D array
        avg=np.reshape(avg, np.size(avg), order='F') #make it 1D
        avg -= np.median(avg)
        
        arr = avg#np.append(np.zeros(500),avg)
        fft = np.fft.fft(arr)

        powspec = (abs(fft))**2

        xaxis = np.arange(len(fft)) - 0.5*len(fft)
        bad = xaxis[np.where(powspec > (np.median(powspec)+3.*np.std(powspec, ddof=1)))]
        bad = bad[np.where((bad > 0) & (bad < np.max(xaxis)))]

        plt.figure(num=1, figsize=(17, 5), dpi=100)
        plt.plot(xaxis, powspec, 'b-')
        #bad = [4,20, 28, 36, 44]
        plt.vlines(bad, 0, np.max(powspec), colors='r')
        #plt.yscale('log')
        #plt.xlim(0,104)
        plt.xlim(xmin=-1)
        plt.title('frame '+str(i))#+', col '+str(col))
        plt.show()

            #pdb.set_trace()


def clean(frame):
    #remove bad frequencies

    avg = [np.median(frame[:,   :32], 1)]#, 
#           np.median(frame[:, 32:64], 1), 
#           np.median(frame[:, 64:  ], 1) ] #a 2D array
    avg=np.reshape(avg, np.size(avg), order='F') #make it 1D
    avg -= np.median(avg)

    #arr = avg#np.append(np.zeros(500),avg)
    fft = np.fft.fft(avg)

    powspec = (abs(fft))**2

    xaxis = np.arange(len(fft)) - 0.5*len(fft)
    #np.ravel converts the array to 1d. I dunno why it wasn't 1D to begin with.
    bads = np.ravel(np.where(powspec > (np.median(powspec)+3.*np.std(powspec, ddof=1))))
    #print bads
    #bads = bads[np.where((bads > 0) & (bads < np.max(xaxis)))]
    #pdb.set_trace()
    if bads[0] == 0: bads=bads[1:] #remove first element
    if bads[0] == 1: bads=bads[1:] #remove first element
    if bads[-1] == len(powspec)-1: bads=bads[:-1] #remove last element

    if 1:#show frequency channels that will be nulled
        plt.figure(num=1, figsize=(17, 5), dpi=100)
        plt.plot(xaxis, powspec)
        plt.vlines(xaxis[bads], 0, 1.1*np.max(powspec), colors='r')
        #plt.vlines(-1*bads+len(fft)/2, 0, np.max(powspec), colors='r')
        plt.xlim(-1.*len(fft)/2, len(fft)/2)
        plt.title('Power Spectrum')
        plt.xlabel('Channel Number')
        plt.ylabel('Power [arbitrary units]')
        plt.show()
    
#    for bad in bads:
#        fft[   bad ] = 0
#        fft[-1*bad + len(fft)] = 0

    fft[bads] = 0
        #fft[-1*bads + len(fft)] = 0

    #inverse fourier transform
    cleaned = np.fft.ifft(fft)#[500:]
    diff = avg - np.real(cleaned)
    pdb.set_trace()
    cleaned = np.zeros(np.shape(frame))
    for y in range(np.shape(frame)[0]):
        for x in range(np.shape(frame)[1]/32):
            cleaned[y, x*32:(x+1)*32] = frame[y, x*32:(x+1)*32] - diff[x*np.shape(frame)[0]+y]
    #pdb.set_trace()
    if 1:
        #plt.figure(num=1, figsize=(5, 5), dpi=100)
        plt.figure(num=1, figsize=(17, 5), dpi=100)
        plt.subplot(131)
        mymin = np.min(frame)
        mymax = np.max(frame)
        plt.imshow(frame, interpolation='none', vmin=mymin, vmax=mymax)
        plt.title("Before")
        
        plt.subplot(132)
        plt.imshow(cleaned, interpolation='none', vmin=mymin, vmax=mymax)
        plt.title("After")
        
        plt.subplot(133)
        
        #create an image equivalent to what has been subtracted off
        for q in range(np.shape(frame)[1]/32):
            if q==0:
                correction = np.transpose(np.tile(diff[ : 1*np.shape(frame)[0]], (32,1)))
            else:
                #pdb.set_trace()
                strip = np.transpose(np.tile(diff[q*np.shape(frame)[0] : 
                                                  (q+1)*np.shape(frame)[0]], (32,1))) 
                correction = np.append(correction, strip, axis=1)

        plt.imshow(correction, interpolation='none', cmap='gray')
        plt.colorbar()
        plt.title("Correction")
        plt.show()
    return cleaned

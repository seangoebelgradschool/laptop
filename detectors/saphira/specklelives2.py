#updated 12/6/15
#Checks specklelives. Reads decubed/aligned images. Subtracts
# adjacent images to see
# how long it takes for speckles to change by ___ fraction of core
# flux.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
#import scipy.ndimage.interpolation
import pdb

def speckles():
    filename = '150729_230246_cleanedaligned.fits'
    dir = 'images/'

    img = pyfits.getdata(dir+filename)

    #find center of image
    im_avg_orig = np.sum(img, 0) / np.shape(img)[0]
    im_avg = im_avg_orig - np.sort(im_avg_orig.flatten())[np.size(im_avg_orig)-30]
    im_avg[np.where(im_avg < 0)] = 0
    
    loc = np.where(im_avg < np.sort(im_avg.flatten())[np.size(im_avg)-30])
    x0 = np.sum(np.sum(im_avg, 0)*range(np.shape(im_avg)[1])) / np.sum(im_avg)
    y0 = np.sum(np.sum(im_avg, 1)*range(np.shape(im_avg)[0])) / np.sum(im_avg)

    if 0:
        plt.imshow(im_avg, interpolation='none')
        print x0, y0
        plt.plot([x0], [y0], 'x')
        plt.show()
    
    #calculate mask around center
    mask = np.zeros(np.shape(im_avg))
    noisemask = np.zeros(np.shape(im_avg))
    mask[:,:] = False
    noisemask[:,:] = False
    for x in range(np.shape(mask)[1]):
        for y in range(np.shape(mask)[0]):
            r = np.sqrt((x-x0)**2 + (y-y0)**2)
            if ((r < 30) & (r>5)): #annulus between 5 and 30 px of psf
                mask[y,x] = True

    noisemask[:, 32*4:] = True
                
    maskloc = np.where(mask == True)
    noisemaskloc = np.where(noisemask == True)

    n_avgs = 1000

    separations = np.arange(50)+1
    stddev_storage = np.zeros(len(separations))
    noise_stddev_storage = np.zeros(len(separations))

    frames = (np.round(np.linspace(0, np.shape(img)[0]-np.max(separations)-1,
                                      n_avgs))).astype(int)
    frames = range(np.shape(img)[0]-np.max(separations))
    n_avgs = len(frames)
    
    #calculate stddev of pixels within mask for various frame deltas
    for i in frames:
        if i%(len(frames)/100)==0: #update status
            print str(int(round(float(i)/np.max(frames)*100.)))+"% complete."
        
        darkloc = np.where(img[i][maskloc] <
                           np.sort(img[i][maskloc])[0.5*len(img[i][maskloc])])
        noisedarkloc = np.where(img[i][noisemaskloc] <
                           np.sort(img[i][noisemaskloc])[0.5*len(img[i][noisemaskloc])])
        #print np.sort(img[i][maskloc])[0.5*len(img[i][maskloc])]
        #print np.max(img[i][maskloc][darkloc])
        #print np.sum(mask), np.shape(maskloc), np.shape(darkloc)
        #pdb.set_trace()

        for sep in separations:
            #printi, i+sep, sep-1
            ccds = img[i] - img[i+sep]
            stddev_storage[sep-1] += np.std(ccds[maskloc][darkloc], ddof=1)
            noise_stddev_storage[sep-1] += np.std(ccds[noisemaskloc][noisedarkloc], ddof=1)

            if sep == 0:
                plt.figure(num=1, figsize=(10, 5), dpi=100) 
                #plt.subplot(141)
                #plt.imshow(img[i], interpolation='none', vmin=0, vmax=1e3)

                #plt.subplot(142)
                #plt.imshow(img[i+sep], interpolation='none', vmin=0, vmax=1e3)

                plt.subplot(121)
                #ccds[maskloc] = 0
                plt.imshow(ccds, interpolation='none', vmin=-15, vmax=15)
                plt.title((np.std(ccds[maskloc], ddof=1)))

                plt.subplot(122)
                plt.imshow(img[i] - img[i+sep+1], interpolation='none',
                           vmin=-15, vmax=15)
                plt.title((np.std((img[i] - img[i+sep+1])[maskloc], ddof=1)))
                plt.show()
            #pdb.set_trace()

    stddev_storage /= n_avgs
    noise_stddev_storage /= n_avgs
    
    psf_core = np.max(im_avg_orig)
    
    plt.plot(separations, stddev_storage/psf_core, 'bo')
    plt.plot([0], np.median(noise_stddev_storage)/psf_core, 'ro')
    plt.title(filename[:13]+', PSF Core = '+str(int(round(psf_core)))+' ADU')
    plt.xlabel('Frame delta')
    plt.ylabel('Stddev (fraction of PSF core)')
    plt.ylim(0.8e-3, 1.6e-3)
    plt.xlim(-1, 50)
    plt.show()
    
    #pdb.set_trace()

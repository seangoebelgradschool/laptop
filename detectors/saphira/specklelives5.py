#updated 6/15/17
#Forked from speckelives4. Updated for RR mode. Subtracts
# adjacent images to see
# how long it takes for speckles to change by ___ fraction of core
# flux.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
#import scipy.ndimage.interpolation
import pdb

def speckles():
    filename = 'pbimage_12:25:45.673185625_p_aligned.fits'
    voltage='-5' #determines which dark to use
    darkmode='F' #Are we running on dark images or data images?
    aomode='F' #Does the data have AO corrections?
    dir = '/media/data/20170531/saphira/processed/'
    framerate = 1680. #Hz

    print "Framerate is assumed to be", framerate, "Hz."
    print "Voltage is assumed to be ", voltage
    if aomode != 'F':
        print "Working with AO on."
    else:
        print "Working with no AO."

    #0.0107''/pixel plate scale
    #How many pixels per lambda/D
    lambda_D_pix = 1.63e-6 / 8.2 * 206265. / .0107 

    img = pyfits.getdata(dir+filename)#[:1000]

    #count how many frames have data
    #realframecount = 0
    #for i in range(np.shape(img)[0]):
    #    if np.max(img[i]) != 0:
    #        realframecount += 1
    
    if darkmode == 'F':
        print "Working on real data."

        #find center of image
        im_avg_orig = np.sum(img, 0) / len(img)
        psf_core = np.max(im_avg_orig) #used to scale flux later
        
        if aomode != 'F':
            im_avg = im_avg_orig - np.sort(im_avg_orig.flatten())[np.size(im_avg_orig)-30]
            im_avg[np.where(im_avg < 0)] = 0
        else:
            im_avg = im_avg_orig - np.median(im_avg_orig)
            im_avg[np.where(im_avg < 0)] = 0
        
        x0 = np.sum(np.sum(im_avg, 0)*range(np.shape(im_avg)[1])) / np.sum(im_avg)
        y0 = np.sum(np.sum(im_avg, 1)*range(np.shape(im_avg)[0])) / np.sum(im_avg)

        if 1:
            plt.imshow(im_avg_orig, interpolation='none')
            print x0, y0
            plt.plot([x0], [y0], 'x')
            plt.show()
    else: #if it is dark data
        print "Operating on dark data."
        x0 = np.shape(img)[2] / 2
        y0 = np.shape(img)[1] / 2

        im_avg = np.sum(img, 0) / len(img)
        psf_core = np.max(im_avg) #used to scale flux later

    #calculate mask around center
    mask = np.zeros(np.shape(im_avg))
   # noisemask = np.zeros(np.shape(im_avg))
    mask[:,:] = False
   # noisemask[:,:] = False
    for x in range(np.shape(mask)[1]):
        for y in range(np.shape(mask)[0]):
            r = np.sqrt((x-x0)**2 + (y-y0)**2)
            #select annulus between 2 and 8 lambda/D
            if ((r < 8.*lambda_D_pix) & (r>2.*lambda_D_pix)): 
                mask[y,x] = True
    if aomode=='F':
        mask[:,:12] = False

    maskloc = np.where(mask == True)
   # noisemaskloc = np.where(noisemask == True)

    separations = np.append(np.arange(50)+1, np.arange(52,260,4))
    stddev_storage_1 = np.zeros(len(separations)) #10% brightness
    stddev_storage_2 = np.zeros(len(separations)) #80% brightness
    n_entries = np.zeros(len(separations)) #how many data points are in each stddev_storage bin?
   # noise_stddev_storage = np.zeros(len(separations))

    #n_avgs = 1000
    #frames = (np.round(np.linspace(0, np.shape(img)[0]-np.max(separations)-1,
    #                               n_avgs))).astype(int)
    frames = range(np.shape(img)[0]-np.max(separations))
    #n_avgs = len(frames)
    
    #calculate stddev of pixels within mask for various frame deltas
    for i in frames:
        if i%(round(np.max(frames)/100.))==0: #update status
            print str(int(round(float(i)/np.max(frames)*100.)))+"% complete."

        #if np.max(img[i]) == 0: continue #image is blank, so skip to next one
            
        darkloc_1 = np.where(img[i][maskloc] <
                           np.sort(img[i][maskloc])[0.1*len(img[i][maskloc])])
#should this be a >?
        darkloc_2 = np.where(img[i][maskloc] <
                           np.sort(img[i][maskloc])[0.8*len(img[i][maskloc])])
      #  noisedarkloc = np.where(img[i][noisemaskloc] <
      #                     np.sort(img[i][noisemaskloc])[0.5*len(img[i][noisemaskloc])])
        #print np.sort(img[i][maskloc])[0.5*len(img[i][maskloc])]
        #print np.max(img[i][maskloc][darkloc])
        #print np.sum(mask), np.shape(maskloc), np.shape(darkloc)
        #pdb.set_trace()

        for j in range(len(separations)):
            sep = separations[j]
            #printi, i+sep, sep-1

            #if np.max(img[i+sep]) == 0: continue #don't bother with this, skip to next
            
            ccds = img[i] - img[i+sep]
            stddev_storage_1[j] += np.std(ccds[maskloc][darkloc_1], ddof=1)
            stddev_storage_2[j] += np.std(ccds[maskloc][darkloc_2], ddof=1)
            n_entries[j] += 1
           # noise_stddev_storage[sep-1] += np.std(ccds[noisemaskloc][noisedarkloc], ddof=1)
            

            if 0:#sep == 1:
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

    stddev_storage_1 /= n_entries
    stddev_storage_2 /= n_entries
    #stddev_storage_1[0] /= np.sqrt(3./2.) #assume read noise limited
    #stddev_storage_2[0] /= np.sqrt(3./2.) #assume read noise limited
    #stddev_storage_1 /= psf_core #rescale to fraction of PSF brightness  
    #stddev_storage_2 /= psf_core #rescale to fraction of PSF brightness
   
   # noise_stddev_storage /= n_avgs

    time_axis = separations/framerate

    if darkmode == 'T':
        np.savetxt('noise'+voltage+'.txt', [np.median(stddev_storage_1), 
                                            np.median(stddev_storage_2)])
        print "dark data saved as", 'noise'+voltage+'.txt'
    else:
        noise1, noise2 = np.loadtxt('noise'+voltage+'.txt')
        print "referencing dark data from: ", 'noise'+voltage+'.txt'

        #subtract noise in quadrature
        stddev_storage_1 = np.sqrt(stddev_storage_1**2 - noise1**2) 
        stddev_storage_2 = np.sqrt(stddev_storage_2**2 - noise2**2)

    #fit a polynomial to the variance data
    coeffs_1 = np.polyfit(time_axis[:10], stddev_storage_1[:10], 2) #quadratic fit
    coeffs_2 = np.polyfit(time_axis[:10], stddev_storage_2[:10], 2) #quadratic fit
    fit_1 = np.poly1d(coeffs_1)
    fit_2 = np.poly1d(coeffs_2)
    noisefit_1 = fit_1(time_axis)
    noisefit_2 = fit_2(time_axis)

    #stddev_storage_1 -= coeffs_1[2] #subtract y-intercept
    #stddev_storage_2 -= coeffs_2[2] #subtract y-intercept

    plt.figure(num=1, figsize=(12, 5), dpi=100) 
    plt.plot(time_axis, stddev_storage_1, 'bo', label='10% dimmest pixels')
    plt.plot(time_axis, stddev_storage_2, 'ro', label='80% dimmest pixels')
    plt.legend(loc=2)

    #plt.plot(time_axis[:10], stddev_storage_1[:10], 'go') #show which points are being fit to
    #plt.plot(time_axis[:10], stddev_storage_2[:10], 'mo') #show which points are being fit to

    #plt.plot(time_axis[:10], noisefit_1[:10], 'g-')
    #plt.plot(time_axis[:10], noisefit_2[:10], 'm-')
    
    #Title plot
    if ('12:15' in filename) or ('12:16' in filename) or ('12:17' in filename):
        mytitle="AO188 + Extreme AO"
    elif ('12:19' in filename) or ('12:20' in filename) or ('12:21' in filename):
        mytitle="AO188 Only"
    elif ('12:25' in filename) or ('12:26' in filename):
        mytitle="No AO Correction"
    else:
        mytitle = raw_input("Enter a plot title. ")    
    plt.title(mytitle)#+', Median PSF Core = '+str(int(round(psf_core)))+' ADU')
        
    plt.xlabel('Seconds')
    plt.ylabel('Standard deviation (ADU)')
    #plt.ylim(0, np.max(stddev_storage_2) + 0.1*np.ptp(stddev_storage_2) )
    plt.xlim(0, 0.1)#np.max(separations)/framerate)
    
    #plt.text(time_axis[19] + .006, stddev_storage_1[19]+200,'10% dimmest pixels: y = ' + str(coeffs_1[0])[:6]+'x^2 + ' + str(coeffs_1[1])[:6] + 'x + ' + str(coeffs_1[2])[:4], color='g')
    #plt.text(time_axis[19] + .006, stddev_storage_2[19]+200, '80% dimmest pixels: y = ' + str(coeffs_2[0])[:6]+'x^2 + ' + str(coeffs_2[1])[:6] + 'x + ' + str(coeffs_2[2])[:4], color='m')
    #plt.text(time_axis[19] + .006, stddev_storage_1[19]+200,'10% dimmest pixels', color='g')
    #plt.text(time_axis[19] + .006, stddev_storage_2[19]+200,'80% dimmest pixels', color='m')

    plt.show()

    pdb.set_trace()

#/usr/bin/env python

#updated 6/30/15

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation
import pdb

#exclude bad frames from centering

def strfcn():
    dir = '' #../logs/'
    filename = 'saphira_10:37:22.251925018_cleanedaligned.fits'
    img = pyfits.getdata(dir+filename)[350:1600, :,:]
    pdb.set_trace()
    #Calculate approximate center of PSF and find avg intensities
    im_coadd = np.sum(img,0)
    selection = im_coadd > np.sort(im_coadd.flatten())[np.size(im_coadd)*0.983]
    im_coadd -= (np.min((im_coadd*selection)[np.where(im_coadd*selection>0)]))

    #set outside edges to be 0, leaving only the inner ninth of image
    im_coadd[:np.shape(im_coadd)[0]/3]      = 0
    im_coadd[ np.shape(im_coadd)[0] /3*2 :] = 0
    im_coadd[:np.shape(im_coadd)[1]/3]      = 0
    im_coadd[ np.shape(im_coadd)[1]/3*2 :]  = 0

    x0 = np.sum(np.sum(im_coadd*selection, 0)*range(np.shape(im_coadd)[1])) / \
        np.sum(im_coadd*selection)
    y0 = np.sum(np.sum(im_coadd*selection, 1)*range(np.shape(im_coadd)[0])) / \
        np.sum(im_coadd*selection)

    im_coadd = np.sum(img,0)#fix things for intensity referencing
    scalemin = np.sort(im_coadd.flatten())[np.size(im_coadd)*0.03]
    scalemax = np.sort(im_coadd.flatten())[np.size(im_coadd)*0.995]

    img_var = np.zeros((20,20))
    img_time = np.zeros((20,20))
    img_counts = np.zeros((20,20))

    for y in range(0, np.shape(img)[1], 1):
        print y
        for x in range(0, np.shape(img)[2], 1):
            #x axis
            r = round(np.sqrt((x-x0)**2 + (y-y0)**2))
            #y axis. rescale to 0-20 range
            I = round((im_coadd[y,x] - scalemin) / ((scalemax - scalemin) / np.shape(img_var)[0]))
            if I > 19: I=19 #who cares about the PSF core?
            if r > 19: continue

            #y=int(round(np.random.uniform(low=np.shape(img)[1]/3, high=np.shape(img)[1]/3*2)))
            #x=int(round(np.random.uniform(low=np.shape(img)[2]/3, high=np.shape(img)[2]/3*2)))
            pix = img[:, y, x]
        
            N = np.size(pix)
        
            strfcns = np.array([])
            ts = np.array([])
            for t in range(1, 100, 1): #finer resolution
                i = np.arange(0, N-t)
                sum = 1./(N-t) * np.sum((pix[i] - pix[i+t])**2)
                strfcns = np.append(strfcns, sum)
                ts = np.append(ts, t)
            for t in range(100, 400, 4): #coarser resolution
                i = np.arange(0, N-t)
                sum = 1./(N-t) * np.sum((pix[i] - pix[i+t])**2)
                strfcns = np.append(strfcns, sum)
                ts = np.append(ts, t)

            strfcns -= np.median((np.sort(strfcns[0:10]))[0:4]) #subtract off y intercept
            #first guess
            final = np.median(strfcns[ts>200])
            ev_time = ts[np.where((abs(strfcns[ts<200] - 0.5*final) == 
                                   np.min(abs(strfcns[ts<200] - 0.5*final))))]

            #if there are a bunch of higher points left of "ev timescale"
            exclusion = 20 #points to exclude from evolution timescale calculation
            while np.size(np.where(strfcns[ts<ev_time] > strfcns[np.where(ts==ev_time)])) > 20:
                ev_time = ts[np.where((abs(strfcns[ts<(200-exclusion)] - 0.5*final) == 
                                       np.min(abs(strfcns[ts<(200-exclusion)] - 0.5*final))))]
                exclusion += 20

            if 3*ev_time < np.max(ts): #should be true
                #refine evolution timescale
                final = np.median(strfcns[ts>3.*ev_time])
                ev_time = ts[np.where((abs(strfcns[ts<(3.*ev_time)] - 0.5*final) == 
                                       np.min(abs(strfcns[ts<(3.*ev_time)] - 0.5*final))))]
                #if there are a bunch of higher points left of "ev timescale"
                exclusion = 20
                while np.size(np.where(strfcns[ts<ev_time] > strfcns[np.where(ts==ev_time)])) > 20:
                    ev_time = ts[np.where((abs(strfcns[ts<(200-exclusion)] - 0.5*final) == 
                                           np.min(abs(strfcns[ts<(200-exclusion)] - 0.5*final))))]
                    exclusion += 20

            if (x==y) & (x%5==0):
                plt.plot(ts, strfcns, 'bo')
                plt.title("Pixel position:" + 
                          ' x=' + str(int(round(float(x)/np.shape(img)[2]*100.))) + '%, '
                          ' y=' + str(int(round(float(y)/np.shape(img)[1]*100.))) + '%')
                plt.xlabel('t (frame difference)')
                plt.ylabel('Structure Function')
               # plt.axhline(np.var(pix, ddof=1))
               # plt.annotate('pixel variance', [0.75*np.max(ts), np.var(pix, ddof=1)], color='b')

                plt.axhline(final, color='r')
                plt.annotate('final', [0.75*np.max(ts), final], color='r')
                
                plt.axvline(ev_time, color='r')
                plt.annotate('Evolution time', [ev_time, np.max(strfcns)], color='r')
                plt.show()
                #pdb.set_trace()

            if (I== 20) or (r == 20):
                print "about to crash!"
                pdb.set_trace()
            if (np.size(I) != 1) or (np.size(r) != 1) or (np.size(ev_time) != 1):
                print "about to crash!"
                pdb.set_trace()

            img_time[I,r] += ev_time
            img_var[I,r] += np.var(strfcns[ts>3.*ev_time], ddof=1)
            img_counts[I,r] += 1

    img_time /= img_counts
    img_var /= img_counts

    plt.figure(1, figsize=(15, 5), dpi=100) 
    plt.subplot(131)
    plt.imshow(img_time, interpolation='none')
    plt.title('Evolution timescale')
    plt.xlabel('Radius (px)')
    plt.ylabel('Intensity (arb. scaling)')

    plt.subplot(132)
    plt.imshow(img_counts, interpolation='none')
    plt.title('Pixels included in Calculation')
    plt.xlabel('Radius (px)')
    plt.ylabel('Intensity (arb. scaling)')

    plt.subplot(133)
    plt.imshow(img_var, interpolation='none', vmin=np.nanmin(img_var), 
               vmax=np.sort(img_var[np.isfinite(img_var)])[np.size(img_var[np.isfinite(img_var)])-3])
    plt.title('Image Variance')
    plt.xlabel('Radius (px)')
    plt.ylabel('Intensity (arb. scaling)')

    plt.show()

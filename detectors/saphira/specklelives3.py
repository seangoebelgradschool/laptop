#Calculates speckle variances using the technique described in
# Macintosh et al 2011
#
#Last updated 3/24/16

import pyfits
import matplotlib.pyplot as plt
import numpy as np
#import scipy.ndimage.interpolation
import pdb

def speckles():
    filename = '150729_230116_cleanedaligned.fits'
    dir = 'images/'

    img = pyfits.getdata(dir+filename)

    #find center of image
    im_avg_orig = np.sum(img, 0) / np.shape(img)[0]
    im_avg = im_avg_orig - np.sort(im_avg_orig.flatten())[np.size(im_avg_orig)-30]
    im_avg[np.where(im_avg < 0)] = 0
    
    loc = np.where(im_avg < np.sort(im_avg.flatten())[np.size(im_avg)-30])
    x0 = np.sum(np.sum(im_avg, 0)*range(np.shape(im_avg)[1])) / np.sum(im_avg)
    y0 = np.sum(np.sum(im_avg, 1)*range(np.shape(im_avg)[0])) / np.sum(im_avg)

    #crop = img[:, 34:46  , 30:45]

    #print "creating radius image"
    radius_img = np.zeros(np.shape(img[0]))
    for y in range(np.shape(radius_img)[0]):
        for x in range(np.shape(radius_img)[1]):
            radius_img[y,x] = np.sqrt((x-x0)**2 + (y-y0)**2)

    #print "done"

    loc = np.where((radius_img < 35) & (radius_img > 20))

    #plt.imshow(radius_img)
    #plt.colorbar()
    #plt.show()

    variances = np.zeros(len(img))#2000)
    for i in range(len(variances)):
        if i%100 == 0: #update status
            print str(int(round(float(i)/float(len(variances))*100.)))+"% complete"
            
        if i == 0: avg = img[0][loc]
        else:
            avg = avg*(1.-1./float(i+1)) + img[i][loc]/float(i+1)

        variances[i] = np.var(avg, ddof=1)

        if 0:#i%200==0:
            plt.imshow(avg, interpolation='none')
            plt.colorbar()
            plt.show()
        
    plt.loglog(variances, '.')
    plt.ylabel("variance [adu^2]")
    plt.xlabel("frame number")
    plt.title("Pixel decorrelation")

    #overplot fit
    xaxis = np.arange(len(variances)+1)
    plt.plot(xaxis, 53. * xaxis**(-0.13) )
    
    plt.show()

    pdb.set_trace()

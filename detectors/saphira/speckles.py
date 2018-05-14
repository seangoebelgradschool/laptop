#/usr/bin/env python

#updated 8/22/15
#Takes image that has been de-cubed. Aligns and removes hot pixels. Saves
# image for additional analysis.


import pyfits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation
import pdb

WHAT THE HECK IS GOING ON WITH 'SUM'?


def speckles():
    dir = '' #../logs/'
    filename = 'saphira_10:37:22.251925018_cleaned.fits'
    img = pyfits.getdata(dir+filename)
    
    #Calculate approximate center of PSF
    im_coadd = np.sum(img,0)
    selection = im_coadd > np.sort(im_coadd.flatten())[np.size(im_coadd)*0.983]

    hotpixelocs = find_hot_pixels(im_coadd*selection)
    selection[hotpixelocs] = False

    im_coadd -= (np.min((im_coadd*selection)[np.where(im_coadd*selection>0)]))

    #set outside edges to be 0, leaving only the inner ninth of image
    im_coadd[:np.shape(im_coadd)[0]/3]      = 0
    im_coadd[ np.shape(im_coadd)[0] /3*2 :] = 0
    im_coadd[:np.shape(im_coadd)[1]/3]      = 0
    im_coadd[ np.shape(im_coadd)[1]/3*2 :]  = 0

    x0 = sum(np.sum(im_coadd*selection, 0)*range(np.shape(im_coadd)[1])) / \
        np.sum(im_coadd*selection)
    y0 = sum(np.sum(im_coadd*selection, 1)*range(np.shape(im_coadd)[0])) / \
        np.sum(im_coadd*selection)

    #Recenter PSFs to reduce tip/tilt
    for i in range(350, np.shape(img)[0]):
        if i % 100 ==0:
            print int(np.round(np.float(i)/np.shape(img)[0]*100)), "% done."
        im = np.copy(img[i,:,:])

        selection = im > np.sort(im.flatten())[np.size(im)*0.989]
 
        selection[hotpixelocs] = False

        #set outside edges to be 0, leaving only the inner ninth of image
        selection[:np.shape(im)[0]/3]      = False
        selection[ np.shape(im)[0] /3*2 :] = False
        selection[:np.shape(im)[1]/3]      = False
        selection[ np.shape(im)[1] /3*2 :] = False

        im -= (np.min((im * selection)[np.where(im * selection > 0)])) #background subtract

        x1 = sum(np.sum(im*selection, 0)*range(np.shape(im)[1]))/np.sum(im*selection)
        y1 = sum(np.sum(im*selection, 1)*range(np.shape(im)[0]))/np.sum(im*selection)

        #check shifting
        if 0:
            plt.figure(1, figsize=(15, 5), dpi=100) 
            plt.subplot(131)
            plt.imshow(crop_avg, interpolation='none')
            plt.plot([x0] , [y0], 'x')
        
            plt.subplot(132)
            plt.imshow(crop*selection, interpolation='none')
            plt.plot([x1] , [y1], 'x')
        
            #crop = np.roll(crop, int(round(x0-x1)), axis=1)
            #crop = np.roll(crop, int(round(y0-y1)), axis=0)
            
            crop = scipy.ndimage.interpolation.shift(crop, [y0-y1, x0-x1], mode='wrap')
            
            plt.subplot(133)
            plt.imshow(crop, interpolation='none')
            plt.plot([x0] , [y0], 'x')
            plt.show()

        im_edit = hot_pixel_clean(hotpixelocs, img[i,:,:])
        
        img[i,:,:] = scipy.ndimage.interpolation.shift(im_edit, [y0-y1, x0-x1], mode='wrap')

    newfilename = filename[:filename.find('.fits')] + 'aligned.fits'
    pyfits.writeto(newfilename, img, clobber='true') #save file

def find_hot_pixels(img):
#Looks at an image and returns locations of pixels that are surrounded by 0s

    sum = np.roll(img, 1, axis=0) + np.roll(img, -1, axis=0) + \
          np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1)
    loc = np.where((sum == 0) & (img != 0))

    return loc

def hot_pixel_clean(locs, img):
    #given the locations of hot pixels in a 2D, interpolates to remove them
    for j in range(np.shape(locs)[1]):

        #vast majority of cases
        if (locs[0][j] > 0) & (locs[0][j] < np.shape(img)[0] -1) & \
           (locs[1][j] > 0) & (locs[1][j] < np.shape(img)[1] -1) :
            pix = np.array([ img[locs[0][j]-1 , locs[1][j] ], #y-1
                             img[locs[0][j]+1 , locs[1][j] ], #y+1
                             img[locs[0][j] , locs[1][j]-1 ], #x-1
                             img[locs[0][j] , locs[1][j]+1 ], #x+1
                             img[locs[0][j]-1 , locs[1][j]-1 ], #y-1, x-1
                             img[locs[0][j]-1 , locs[1][j]+1 ], #y-1, x+1
                             img[locs[0][j]+1 , locs[1][j]-1 ], #y+1, x-1
                             img[locs[0][j]+1 , locs[1][j]+1 ] ]) #y+1, x+1
            dist = np.array([1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2) ])
            

        else: #when a hot pixel is on the edge or corner of the frame
            pix = np.array([])
            dist = np.array([])

            #if not on bottom of image
            if locs[0][j] != 0: 
                pix = np.append(pix, img[locs[0][j]-1 , locs[1][j] ])
                dist = np.append(dist, 1)

            #if not on top of image
            if locs[0][j] != np.shape(img)[0]-1: 
                pix = np.append(pix, img[locs[0][j]+1 , locs[1][j] ])
                dist = np.append(dist, 1)

            #if not on left edge of image
            if locs[1][j] != 0: 
                pix = np.append(pix, img[locs[0][j] , locs[1][j]-1 ])
                dist = np.append(dist, 1)

            #if not on right edge of image
            if locs[1][j] != np.shape(img)[1]-1: 
                pix = np.append(pix, img[locs[0][j] , locs[1][j]+1 ])
                dist = np.append(dist, 1)

            #if not bottom left corner of image
            if (locs[0][j] != 0) & (locs[1][j] != 0) : 
                pix = np.append(pix, img[locs[0][j]-1 , locs[1][j]-1 ])
                dist = np.append(dist, np.sqrt(2))

            #if not top left corner
            if (locs[0][j] != np.shape(img)[0]-1) & (locs[1][j] != 0): 
                pix = np.append(pix, img[locs[0][j]+1 , locs[1][j]-1 ])
                dist = np.append(dist, np.sqrt(2))
            
            #if not bottom right corner
            if (locs[0][j] != 0) & (locs[1][j] != np.shape(img)[1]-1): 
                pix = np.append(pix, img[locs[0][j]-1 , locs[1][j]+1 ])
                dist = np.append(dist, np.sqrt(2))

            #if not top right corner
            if (locs[0][j] != np.shape(img)[0]-1) & (locs[1][j] != np.shape(img)[1]-1):
                pix = np.append(pix, img[locs[0][j]+1 , locs[1][j]+1 ])
                dist = np.append(dist, np.sqrt(2))

        #print
        #print "before:"
        #print img[locs[0][j]-1 : locs[0][j]+2 , locs[1][j]-1 : locs[1][j]+2]

        #interpolate pixel
        img[locs[0][j], locs[1][j]] = np.sum(pix * 1./dist) / np.sum(1./dist)

        #print "after:"
        #print img[locs[0][j]-1 : locs[0][j]+2 , locs[1][j]-1 : locs[1][j]+2]
        #print
    return img

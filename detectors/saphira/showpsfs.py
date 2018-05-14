#plots speckle images for use in figures

import pyfits
import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.colors import LogNorm

def show():
    dir = 'images/'
    images = ['saphira_14:46:36.269917196_cleanedaligned.fits' , #pyr
              'saphira_14:46:51.541667336_cleanedaligned.fits' , #ao188
              'saphira_14:47:44.962387627_cleanedaligned.fits'] #open

    titles = ['AO188 + Extreme AO',
              'AO188 Only',
              'No AO']
    
    #0.0107''/pixel plate scale
    #How many pixels per lambda/D
    lambda_D_pix = 1.63e-6 / 8.2 * 206265. / .0107 

    for j in range(len(images)):
        img = pyfits.getdata(dir+images[j])

        #compute average image
        n_data_frames = 0
        for i in range(np.shape(img)[0]):
            if np.max(img[i]) != 0:
                n_data_frames += 1

        im_coadd = np.sum(img, axis=0) / n_data_frames #compute average
        im_avg = np.copy(im_coadd) #silly python, why must i do this

        #select 29 brightest pixels
        selection = im_coadd > np.sort(im_coadd.flatten())[np.size(im_coadd)-30] 
        im_coadd -= (np.min((im_coadd*selection)[np.where(im_coadd*selection>0)]))

        x0 = np.sum(np.sum(im_coadd*selection, 0)*range(np.shape(im_coadd)[1])) / \
             np.sum(im_coadd*selection)
        y0 = np.sum(np.sum(im_coadd*selection, 1)*range(np.shape(im_coadd)[0])) / \
             np.sum(im_coadd*selection)

        #Figure out circles
        theta = np.linspace(0, 2.*np.pi, 100)
        r1 = 2 * lambda_D_pix
        r2 = 8 * lambda_D_pix
        x1 = r1 * np.cos(theta) + x0
        y1 = r1 * np.sin(theta) + y0
        x2 = r2 * np.cos(theta) + x0
        y2 = r2 * np.sin(theta) + y0


        #PLOT AVERAGE IMAGE
        plt.imshow(im_avg, interpolation='none', norm=LogNorm(), vmin=1)
        plt.plot(x1, y1, color='black') #plot circle
        plt.plot(x2, y2, color='black')#plot circle
        plt.colorbar(shrink=0.8)
        plt.title(titles[j] + ' Coadded Image')
        plt.xlim(0, np.shape(im_avg)[1]-1)
        plt.ylim(0, np.shape(im_avg)[0]-1)
        plt.show()


        #FIGURE OUT 50TH PERCENTILE IMAGE
        n_checks = 100
        relstrehls = np.zeros(n_checks)
        frames = (np.round(np.linspace(0, np.shape(img)[0]-1, n_checks))).astype(int)
        goods = np.array([])

        for i in range(len(frames)):
            if np.max(img[frames[i]]) != 0:
                relstrehls[i] = np.max(img[frames[i], y0-10 : y0+10 , x0-10 : x0+10])
                goods = np.append(goods, i)

        relstrehls = relstrehls[goods.astype(int)] #remove blank frames
        frames = frames[goods.astype(int)] #remove blank frames

        #select 50% percentile
        loc = np.argmax(relstrehls == np.sort(relstrehls)[0.5 * len(relstrehls)])
        pic = img[frames[loc]]

        pic[pic<1] = 1

        #mymin = np.sort(pic.flatten())[np.size(img[0])*.01]
        mymax = np.sort(pic.flatten())[np.size(img[0])*.999]

        plt.imshow(pic, interpolation='none', vmin=3, vmax=mymax, norm=LogNorm())
        plt.plot(x1, y1, color='black') #plot circle
        plt.plot(x2, y2, color='black')#plot circle
        plt.colorbar(shrink=0.8)
        plt.title(titles[j] + ' Typical Single Image')
        if '47:44' in images[j]:
            plt.xlim(0, np.shape(im_avg)[1]-3)
        else:
            plt.xlim(0, np.shape(im_avg)[1]-1)
        plt.ylim(0, np.shape(im_avg)[0]-1)
        plt.show()

    

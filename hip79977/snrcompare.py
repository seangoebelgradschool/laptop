import matplotlib.pyplot as plt
import numpy as np
import pyfits
import os
import pdb

def main():
    dir = 'snrmaps/'
    files = os.listdir(dir)

    for i in range(len(files)):
        print files[i]
        if i==0:
            avgcube = pyfits.getdata(dir+files[i]) #create cube
        else:
            avgcube += pyfits.getdata(dir+files[i]) #add to cube
    avgcube /= len(files)

    #i'm tired, sorry for laziness
    ximg = np.zeros(np.shape(avgcube))
    yimg = np.zeros(np.shape(avgcube))
    for j in range(np.shape(avgcube)[2]): #assumes xsize = ysize
        ximg[:,:,j] = j
        yimg[:,j,:] = j
        
    theta = np.deg2rad(78)
    xshift = 95
    yshift = 100
    b = 2*70
    a = 2*11
    loc = ((yimg - yshift) < (ximg - xshift) * np.tan(theta) + 2*a/np.sin(theta)) & \
          ((yimg - yshift) > (ximg - xshift) * np.tan(theta) - 2*a/np.sin(theta)) & \
          ((yimg - yshift) < (ximg - xshift) * np.tan(theta - np.pi/2.) + 2*b/np.tan(theta)) & \
          ((yimg - yshift) > (ximg - xshift) * np.tan(theta - np.pi/2.) - 2*b/np.tan(theta))
    
    for i in range(len(files)):
        img = pyfits.getdata(dir+files[i]) / avgcube
        img[np.isnan(img)] = 0

        snrs = np.zeros(np.shape(img)[0])
        for z in range(len(snrs)):
            snrs[z] = np.median((img*loc)[np.where(img*loc != False)])
            

        plt.plot(snrs, 'o', label=str(np.median(snrs))[:6])
        plt.legend()
        plt.title(files[i])
        plt.xlabel('Channel Number')
        plt.ylabel('(SNR of Channel) / (Avg SNR of Channel)')
        plt.savefig((files[i]).replace('.fits', '_snrplot.png'), bbox_inches='tight')
        plt.clf()
        #plt.show()

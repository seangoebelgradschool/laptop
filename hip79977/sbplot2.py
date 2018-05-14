import pyfits
import matplotlib.pyplot as plt
#import matplotlib.axes.Axes
import numpy as np
import pdb

def main(save=False):

    junk = np.loadtxt('figs/biggerr/sb_j.txt', dtype='float')
    eloc = np.squeeze(np.where(junk[:,0] < 0))
    wloc = np.squeeze(np.where(junk[:,0] > 0))
    jer = -1 * junk[eloc,0]
    je = junk[eloc,1]
    jeu = junk[eloc,2]
    jwr = junk[wloc,0]
    jw = junk[wloc,1]
    jwu = junk[wloc,2]

    junk = np.loadtxt('figs/biggerr/sb_h.txt', dtype='float')
    eloc = np.squeeze(np.where(junk[:,0] < 0))
    wloc = np.squeeze(np.where(junk[:,0] > 0))
    her = -1 * junk[eloc,0]
    he = junk[eloc,1]
    heu = junk[eloc,2]
    hwr= junk[wloc,0]
    hw = junk[wloc,1]
    hwu = junk[wloc,2]

    junk = np.loadtxt('figs/biggerr/sb_k.txt', dtype='float')
    eloc = np.squeeze(np.where(junk[:,0] < 0))
    wloc = np.squeeze(np.where(junk[:,0] > 0))
    ker = -1 * junk[eloc,0]
    ke = junk[eloc,1]
    keu = junk[eloc,2]
    kwr = junk[wloc,0]
    kw = junk[wloc,1]
    kwu = junk[wloc,2]

    #pdb.set_trace()
    
    plt.figure(figsize=(7.5, 10), dpi=100)

    #J BAND
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.errorbar(jer, je, yerr=jeu, fmt='bo', label='East')
    plt.errorbar(jwr, jw, yerr=jwu, fmt='ro', label='West')
    plt.title(r'$J$'+ ' Band Surface Brightness of Disk Spine', fontsize=20)
    plt.ylabel('Magnitude per arcsec'+r'$^2$', fontsize=16)
    plt.xlabel('R [arcsec]', fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.set_autoscale_on(False)
    plt.xlim(0.1, 1.1)
    plt.ylim(17, 10)

    #H BAND
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.errorbar(her, he, yerr=heu, fmt='bo', label='East')
    plt.errorbar(hwr, hw, yerr=hwu, fmt='ro', label='West')
    plt.title(r'$H$'+ ' Band Surface Brightness of Disk Spine', fontsize=20)
    plt.ylabel('Magnitude per arcsec'+r'$^2$', fontsize=16)
    plt.xlabel('R [arcsec]', fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.set_autoscale_on(False)
    plt.xlim(0.1, 1.1)
    plt.ylim(17, 10)

    #K BAND
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.errorbar(ker, ke, yerr=keu, fmt='bo', label='East')
    plt.errorbar(kwr, kw, yerr=kwu, fmt='ro', label='West')
    plt.title(r'$K_p$'+ ' Band Surface Brightness of Disk Spine', fontsize=20)
    plt.ylabel('Magnitude per arcsec'+r'$^2$', fontsize=16)
    plt.xlabel('R [arcsec]', fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.set_autoscale_on(False)
    plt.xlim(0.1, 1.1)
    plt.ylim(17, 10)

    plt.subplots_adjust(left=0.09, bottom=0.07, right=0.97, top=0.96, hspace=0.40)
    
    
    if save:
        outtitle = 'figs/surfacebrightness2.png'
        plt.savefig(outtitle, bbox_inches='tight', dpi=200)
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()

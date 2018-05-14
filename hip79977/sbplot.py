import pyfits
import matplotlib.pyplot as plt
#import matplotlib.axes.Axes
import numpy as np

def main(save=False):

    junk = np.loadtxt('figs/biggerr/sb_j.txt', dtype='float')
    r=junk[:,0]
    j = junk[:,1]
    ju = junk[:,2]

    junk = np.loadtxt('figs/biggerr/sb_h.txt', dtype='float')
    #r=junk[:,0]
    h = junk[:,1]
    hu = junk[:,2]

    junk = np.loadtxt('figs/biggerr/sb_k.txt', dtype='float')
    #r=junk[:,0]
    k = junk[:,1]
    ku = junk[:,2]

    plt.figure(figsize=(7.5, 10), dpi=100)
    for i in range(2): #two plots
    
        ax = plt.gca()
        #ax.invert_yaxis()
        if i==0: #first time around
            plt.subplot(2,1,1)
            plt.title('Surface Brightness of Disk Spine', fontsize=21)
            plt.ylabel('Magnitude per arcsec'+r'$^2$', fontsize=16)
            offset = 0
        else: #relative brightness
            plt.subplot(2,1,2)
            plt.title('Reflectance of Disk Spine', fontsize=21)
            plt.ylabel(r'$\Delta$'+'M(disk - star)', fontsize=16)
            j -= 8.062
            h -= 7.854
            k -= 7.800
            offset = -8
            
        plt.errorbar(r, j, yerr=ju, fmt='bo', label=r'$J$'+' band')
        plt.errorbar(r, h, yerr=hu, fmt='go', label=r'$H$'+' band')
        plt.errorbar(r, k, yerr=ku, fmt='ro', label=r'$K_p$'+' band')
            
        plt.arrow(-0.4, 16+offset, -0.2, 0, head_width=0.7, head_length=0.07, fc='k', ec='k')
        plt.arrow(0.4, 16+offset, 0.2, 0, head_width=0.7, head_length=0.07, fc='k', ec='k')
        plt.text(-0.5, 17+offset, 'East', ha='center', va='center', fontsize=16)
        plt.text(0.5, 17+offset, 'West', ha='center', va='center', fontsize=16)

        plt.xlabel('R [arcsec]', fontsize=16)
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        ax.set_autoscale_on(False)
        plt.xlim(-1.2, 1.2)
        plt.ylim(np.array([18, 8])+offset)
        
    plt.subplots_adjust(left=0.1, bottom=0.07, right=0.99, top=0.96, hspace=0.29)
    if save:
        outtitle = 'figs/surfacebrightness.png'
        plt.savefig(outtitle, bbox_inches='tight', dpi=200)
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()

import numpy as np
import pyfits
import matplotlib.pyplot as plt
#from lmfit.models import LorentzianModel
import scipy.ndimage.interpolation
#from scipy.optimize import curve_fit
#from scipy import asarray as ar,exp
import pdb

def main(save=False):
    old = pyfits.getdata('figs/ao188_hiciao_hip79977.fits')
    new = pyfits.getdata('figs/biggerr/biggerr,2,6,1_collapsed.fits')
    syndisk = pyfits.getdata('figs/biggerr/egrater_0.600000,1.00000,4,-3.50000,112.400,0,0,112.400,84.6000,70,2,0_psfsubcol.fits')

    center = [1000, 1000]
    
    plt.figure(figsize=(5, 7.5), dpi=100)
    plt.subplot(2,1,1)
    plt.imshow(old, interpolation='none', vmin=0, vmax=7e-5)

    plt.arrow(center[0] + 0.5/9.5e-3*np.cos(np.deg2rad(24.6)),
              center[1] + 0.5/9.5e-3*np.sin(np.deg2rad(24.6))-60-10,
              0, 60, head_width=10, head_length=10, linewidth=2, fc='w', ec='w')
    scalebar([880, 1060], distance=66, pixelscale=9.5e-3, linewidth=1.5, fontsize=16)
    compass([920, 930], angle=0, ax=None, length=20, textscale=1.5, fontsize=16, \
            color='white', labeleast=True, linewidth=1.5)
    
    #plt.text(center[1] - x+5, center[0] + x*aspect-6, '(d)', color='black',
    #         bbox=dict(facecolor='white'), ha='left', va='top',size=16*0.9)

    plt.title('Thalmann et al. (2013) Data', fontsize=16)
    #plt.colorbar()
    plt.ylim(900, 1100)
    plt.xlim(850, 1150)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    
    plt.subplot(2,1,2)
    plt.imshow(new - syndisk, \
               interpolation='none', vmin=-1, vmax=6)#, norm=LogNorm())
    #plt.colorbar()
    plt.title("Real Disk - PSF-Subtracted Synthetic Disk", fontsize=16)
    plt.xlim(54, 146)
    plt.ylim(69, 131)
    ax = plt.gca()
    #ax.set_autoscale_on(False)
    #plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    center = [100, 100]
    pixscal=4.5556e-6 * 60*60
    compass([73,80], angle=0, ax=None, length=6, textscale=1.6, fontsize=18, \
            color='white', labeleast=True, linewidth=1.5)
    plt.arrow(center[0] + 0.5/pixscal*np.cos(np.deg2rad(24.6)),
              center[1] + 0.5/pixscal*np.sin(np.deg2rad(24.6))-20-5,
              0, 20, head_width=5, head_length=5, linewidth=1, fc='w', ec='w')
    scalebar([60, 119], distance=66, pixelscale=pixscal, linewidth=1.5, fontsize=14)
    #plt.text(plt.xlim()[0]+2.5, plt.ylim()[1]-2.5,
    #         '(c)', color='black', bbox=dict(facecolor='white'),
    #         ha='left', va='top', size=14)

    #outtitle='figs/realdisk-syndisk.png'


    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95, hspace=0.13)

    outtitle='figs/planet.png'
    if save:
        plt.savefig(outtitle, dpi=150)#, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()
    
    
def compass(origin, angle=0.0, ax=None, length=5, textscale=1.4, fontsize=12, \
            color='white', labeleast=False, **kwargs):

    if ax is None: ax = plt.gca()

    dy =  np.cos(np.deg2rad(angle))*length
    dx = -np.sin(np.deg2rad(angle))*length

    # North
    ax.arrow( origin[0], origin[1], dx, dy, color=color, **kwargs)

    ax.text( origin[0]+textscale*dx, origin[1]+textscale*dy, 'N', \
             verticalalignment='center', horizontalalignment='center', \
             color=color, fontsize=fontsize)

    dy =  np.cos(np.deg2rad(angle+90))*length
    dx = -np.sin(np.deg2rad(angle+90))*length

    # East
    ax.arrow( origin[0], origin[1], dx, dy, color=color, **kwargs)
    if labeleast:
        ax.text( origin[0]+textscale*dx*0.9, origin[1]+textscale*dy, 'E', \
                 verticalalignment='center', horizontalalignment='center', \
                 color=color, fontsize=fontsize) 
    
    
def scalebar(origin, distance=None, pixelscale=0., linewidth=3, fontsize=16, **kwargs):
    dx = 0.5/pixelscale

    ax = plt.gca()
    auto = ax.get_autoscale_on()
    ax.set_autoscale_on(False)

    plt.plot([origin[0], origin[0]+dx], [origin[1], origin[1]], color='white', \
             linewidth=linewidth, clip_on=False, **kwargs)
    plt.text(origin[0]+dx/2, origin[1]+3, '0.5"', horizontalalignment='center', \
             color='white', fontsize=fontsize)

    # reset the autoscale parameter? 

    if distance is not None:
        plt.text(origin[0]+dx/2, origin[1]-3, '{0:d} AU'.format(distance), \
                 horizontalalignment='center', color='white', \
                 verticalalignment='top', fontsize=fontsize)

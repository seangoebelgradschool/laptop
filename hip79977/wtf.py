import matplotlib.pyplot as plt
import numpy as np
import pyfits
import pdb
from matplotlib.colors import LogNorm

def main(pic='syndisk'):

    if pic=='syndisk':
        img = pyfits.getdata('figs/egrater_0.600000,1.00000,4,-3.50000,112.400,0,0,112.400,84.6000,70,2,0.fits')
        mymax = 1e-6
        mymin = 1e-9
    else:
        img = pyfits.getdata('figs/settled2_npca=2,drsub=6,nfwhm=1,rmin=5,rmax=50,meansub,meanadd_collapsed.fits')
        mymax = 5
        mymin = 0
        
    tot_rot=-125.225#112.4

    #rotate image by 90 degrees so N is up
    img = np.rot90(img, k=3)
    tot_rot += 90
    
    plt.imshow(img, interpolation='none', vmin=mymin, vmax=mymax)#, norm=LogNorm())
    #plt.title(title, fontsize=21)
    #plt.xlim(31, 169)
    #plt.ylim(31, 169)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    #plt.plot(center[0], center[1], 'k+', markersize=25, markeredgewidth=3)
    #plt.colorbar()
    #if image=='snr': plt.colorbar()

    #scalebar([110,140], distance=None, pixelscale=pixscal, linewidth=3, fontsize=21)
    compass([70,140], angle=tot_rot, ax=None, length=8, textscale=1.6, fontsize=21, \
            color='green', labeleast=True, linewidth=3)

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

  

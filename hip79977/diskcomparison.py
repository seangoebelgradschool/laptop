import pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation
import pdb

def main(save=False):
    fn_charis_nice = 'figs/pretty/grid_npca=2,nfwhm=150,drsub=2,meansubmeanadd_collapsed.fits'
    fn_charis_ugly = 'figs/ugly/settled2_npca=2,drsub=6,nfwhm=1,rmin=5,rmax=50,meansub,meanadd_collapsed.fits'
    fn_scexao_hiciao = 'figs/sklipnewrawsq.fits'
    fn_ao188_hiciao = 'figs/ao188_hiciao_hip79977.fits'

    charis_nice = pyfits.getdata(fn_charis_nice)
    charis_ugly = pyfits.getdata(fn_charis_ugly)
    scexao_hiciao = pyfits.getdata(fn_scexao_hiciao)
    ao188_hiciao = pyfits.getdata(fn_ao188_hiciao)

    aspect = 2./3. #height / width of images

    plt.figure(figsize=(7.5, 6), dpi=100)
    
#####    
    #CHARIS NICE
    pixscal=4.5556e-6 * 60*60
    center = [100, 100]

    #make r^2 mape for CHARIS Images
    scaling = np.zeros((201, 201))
    for x in range(np.shape(scaling)[1]):
        for y in range(np.shape(scaling)[0]):
            scaling[y, x] = (x-float(center[0]))**2+(y-float(center[1]))**2
    charis_nice *= scaling
            
    crop = 0.65 #fraction of original width
    x = (np.shape(charis_nice)[1] - center[1] -1)*crop

    #rotate image so north is up
    loc = np.isnan(charis_nice)
    charis_nice[loc] = 0
    charis_nice = scipy.ndimage.interpolation.rotate(charis_nice, 234.78, reshape=False)
    charis_nice[loc] = np.nan
    
    plt.subplot(2,2,1)
    plt.imshow(charis_nice, interpolation='none', vmin=0, vmax=3e3)
    plt.title("SCExAO CHARIS Reduction 1")
    
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.xlim(center[1] - x, center[1]+x)
    plt.ylim(center[0] - x*aspect , center[0] + x*aspect)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    scalebar([110, 130], distance=62, pixelscale=pixscal, linewidth=2, fontsize=16)
    compass([75, 120], angle=0, ax=None, length=8, textscale=1.8, fontsize=16, \
            color='white', labeleast=True, linewidth=2)
    plt.text(center[1] - x+1, center[0] + x*aspect-1, '(a)', color='black',
             bbox=dict(facecolor='white'), ha='left', va='top',size=16*0.9)

#####    
    #CHARIS UGLY
    pixscal=4.5556e-6 * 60*60

    crop = 0.45 #fraction of original width
    x = (np.shape(charis_nice)[1] - center[1] -1)*crop

    charis_ugly *= scaling
    
    plt.subplot(2,2,2)
    plt.imshow(charis_ugly, interpolation='none', vmin=0, vmax=3e3, cmap=plt.cm.get_cmap('jet'))
    plt.title("SCExAO CHARIS Reduction 2")
    
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.xlim(center[1] - x, center[1]+x)
    plt.ylim(center[0] - x*aspect , center[0] + x*aspect)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    scalebar([100, 120], distance=62, pixelscale=pixscal, linewidth=2, fontsize=16)
    compass([80, 115], angle=0, ax=None, length=6, textscale=1.6, fontsize=16, \
            color='white', labeleast=True, linewidth=2)
    plt.text(center[1] - x+0.7, center[0] + x*aspect-0.7, '(b)', color='black',
             bbox=dict(facecolor='white'), ha='left', va='top',size=16*0.9)


#####    
    #SCEXAO HICIAO
    plt.subplot(2,2,3)
    plt.imshow(scexao_hiciao, interpolation='none', vmin=0, vmax=5e5)
    plt.title("SCExAO HiCIAO")
    
    ax = plt.gca()
    ax.set_autoscale_on(False)

    crop = 0.7 #fraction of original width
    center = [401, 401]
    x = (np.shape(scexao_hiciao)[1] - center[1] -1)*crop
    plt.xlim(center[1] - x, center[1]+x)
    plt.ylim(center[0] - x*aspect , center[0] + x*aspect)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    scalebar1([500, 520], distance=123, pixelscale=8.3e-3, linewidth=2, fontsize=16)
    compass([280, 490], angle=0, ax=None, length=40, textscale=1.5, fontsize=16, \
            color='white', labeleast=True, linewidth=2)
    plt.text(center[1] - x+5, center[0] + x*aspect-5, '(c)', color='black',
             bbox=dict(facecolor='white'), ha='left', va='top',size=16*0.9)


#####    
    #AO188 HICIAO
    center = [1000, 1000]
    crop = 0.3 #fraction of original width

    #make r^2 map for AO188/HiCIAO Image
    scaling = np.zeros(np.shape(ao188_hiciao))
    for x in range(np.shape(scaling)[1]):
        for y in range(np.shape(scaling)[0]):
            scaling[y, x] = (x-float(center[0]))**2+(y-float(center[1]))**2
    #apply scaling
    ao188_hiciao *= scaling
    
    plt.subplot(2,2,4)
    plt.imshow(ao188_hiciao, interpolation='none', vmin=0, vmax=0.2)
    plt.title("AO188 HiCIAO")
    
    ax = plt.gca()
    ax.set_autoscale_on(False)

    x = (np.shape(ao188_hiciao)[1] - center[1] -1)*crop
    plt.xlim(center[1] - x, center[1]+x)
    plt.ylim(center[0] - x*aspect , center[0] + x*aspect)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    scalebar1([1100, 1125], distance=123, pixelscale=9.5e-3, linewidth=2, fontsize=16)
    compass([870, 1080], angle=0, ax=None, length=45, textscale=1.5, fontsize=16, \
            color='white', labeleast=True, linewidth=2)
    plt.text(center[1] - x+5, center[0] + x*aspect-6, '(d)', color='black',
             bbox=dict(facecolor='white'), ha='left', va='top',size=16*0.9)

#####

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, hspace=0.06, wspace=0.06)
    plt.suptitle('Comparison of HIP 79977 Images', fontsize=18)
    
    if save:
        outtitle='figs/diskcomparison.png'
        plt.savefig(outtitle, bbox_inches='tight', dpi=200)
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

        
def scalebar1(origin, distance=None, pixelscale=0., linewidth=3, fontsize=16, **kwargs):
    dx = 1/pixelscale

    ax = plt.gca()
    auto = ax.get_autoscale_on()
    ax.set_autoscale_on(False)

    plt.plot([origin[0], origin[0]+dx], [origin[1], origin[1]], color='white', \
             linewidth=linewidth, clip_on=False, **kwargs)
    plt.text(origin[0]+dx/2, origin[1]+3, '1.0"', horizontalalignment='center', \
             color='white', fontsize=fontsize)

    # reset the autoscale parameter? 

    if distance is not None:
        plt.text(origin[0]+dx/2, origin[1]-3, '{0:d} AU'.format(distance), \
                 horizontalalignment='center', color='white', \
                 verticalalignment='top', fontsize=fontsize)


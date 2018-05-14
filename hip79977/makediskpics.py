import matplotlib.pyplot as plt
import numpy as np
import pyfits
import pdb
from matplotlib.colors import LogNorm
from PIL import Image, ImageDraw, ImageFont

#indexed from 1
#J : slices 1-5 1.241 um 1.973 px fwhm
#H: slices 8-14 1.620 um 2.592 px
#Kp: slices 16-21 2.103 um 3.344 px
#overall: 1657.768 um 2.64 px fwhm

#J = 1220 +- 213/2
#H = 1630 +- 307/2
#K = 2190 +- 390/2
#waves = np.loadtxt('lowres_wvlh.txt')


def pic(band='none', save='show', image='data', save_band=False, disk='blah', panel=''):

    if band=='none':
        print '''
        Usage: 
        makediskpics.pic(band=band, save=False, image='data', save_band=False, disk='pretty')"
        Options for band: j, h, k, full
        Options for image: data, snr
        Options for disk: pretty, ugly
        Options for save: singlesave, show
        '''
        return
        
    center=[100, 100] #pixel values. good to <0.01 px according to fits header
    pixscal=4.5556e-6 * 60*60 #arcsec/pixel

    if save=='multi':
        fc = 0.5 #font scaling
    else:
        fc=1.
    
    if disk=='pretty':
        folderstub = 'figs/pretty/'
        filestub = folderstub + 'grid_npca=2,nfwhm=150,drsub=2,meansubmeanadd'
    elif disk=='ugly':
        folderstub = 'figs/ugly/'
        filestub = folderstub + 'settled2_npca=2,drsub=6,nfwhm=1,rmin=5,rmax=50,meansub,meanadd'
    else:
        print "Please select disk='pretty' or disk='ugly'"
        return


    #create map showing radius^2 at each point
    scaling = np.zeros((201, 201))
    for x in range(np.shape(scaling)[1]):
        for y in range(np.shape(scaling)[0]):
            scaling[y, x] = (x-float(center[0]))**2+(y-float(center[1]))**2
    
    if image=='data':
        file = filestub+'.fits'
        
        if band.lower()=='full': #collapsed version
            print "Full image"
            img = np.sum(pyfits.getdata(file), 0)
            #img -= np.nanmin(img) #make all values positive
            mymin=-10000
            mymax=60000
            title="Wavelength Collapsed, Radially Scaled"
            outtitle = folderstub + 'hip79977_collapsed.png'
        elif band.lower()=='j': #J band
            print "J band"
            img = np.sum(pyfits.getdata(file)[0:5], 0)
            #mymin=np.nanmin(img)
            #mymax=np.nanmax(img)
            mymin=-1000
            mymax=19000
            title=r'$J$'+" Band, Radially Scaled"
            outtitle = folderstub + 'hip79977_j.png'
        elif band.lower()=='h': #H band
            print "H band"
            img = np.sum(pyfits.getdata(file)[7:14], 0)
            #mymin=np.nanmin(img)
            #mymax=np.nanmax(img)
            mymin=-1000
            mymax=20000
            title=r'$H$'+" Band, Radially Scaled"
            outtitle = folderstub + 'hip79977_h.png'
        elif band.lower()=='k': #k band
            print "K band"
            img = np.sum(pyfits.getdata(file)[15:21], 0)
            #mymin=np.nanmin(img)
            #mymax=np.nanmax(img)
            mymin=-1000
            mymax=12000
            title=r'$K_p$' + " Band, Radially Scaled"
            outtitle = folderstub + 'hip79977_k.png'       

        if save_band==True:
            #save collapsed image for snr analysis
            pyfits.writeto(file[:-5] + '_' + band.lower()+'.fits', img, clobber=True)
            print "Wrote", file[:-5] + '_' + band.lower()+'.fits'
                       
        #apply r^2 scaling to better show outer region of disk
        #Actually apply radial scaling
        img *= scaling

    elif image=='snr':
        
#MASK SNR IMAGES ACCORDING TO NANS IN DATA IMAGES
        
        print "SNR File"
        if band.lower()=='full': #collapsed version
            print "Full image"
            file=filestub + '_collapsed_snr.fits'
            mymin=0
            mymax=8
            title="Wavelength Collapsed, SNR Map"
            outtitle = folderstub + 'hip79977_collapsed_snr.png'
            r_in=12
        elif band.lower()=='j': #J band
            print "J band"
            file=filestub + '_j_snr.fits'
            mymin=0
            mymax=6
            title=r'$J$'+" band, SNR Map"
            outtitle = folderstub + 'hip79977_j_snr.png'
            r_in=6
        elif band.lower()=='h': #H band
            print "H band"
            file=filestub + '_h_snr.fits'
            mymin=0
            mymax=6
            title=r'$H$'+" band, SNR Map"
            outtitle = folderstub + 'hip79977_h_snr.png'
            r_in = 10
        elif band.lower()=='k': #k band
            print "K band"
            file=filestub + '_k_snr.fits'
            mymin=0
            mymax=6
            title=r'$K_p$'+" band, SNR Map"
            outtitle = folderstub + 'hip79977_k_snr.png'
            r_in = 12
        img = pyfits.getdata(file)

        img[scaling < r_in**2] = np.nan
        
    if disk=='pretty':
        tot_rot = -125.22 #degrees

        #rotate image by 90 degrees so N is up
        img = np.rot90(img, k=3)
        tot_rot += 90
        crop = [31, 169]
        annotate = np.array([80, 140])
    else:
        tot_rot = 0
        crop = [53, 147]
        annotate = np.array([80, 125])
    
    #figs/snr_grid_npca=2,nfwhm=150,drsub=2,meansubmeanadd.fitssnrmap.fits'
    
    plt.imshow(img, interpolation='none', vmin=mymin, vmax=mymax)#, norm=LogNorm())
    plt.title(title, fontsize=21*fc, y=1.008)
    plt.xlim(crop)
    plt.ylim(crop)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.plot(center[0], center[1], 'k+', markersize=25*fc, markeredgewidth=3*fc)
    #plt.colorbar()
    if image=='snr': plt.colorbar().ax.tick_params(labelsize=18) #shrink=0.8)

    scalebar(annotate + [20, 5], distance=62, pixelscale=pixscal, linewidth=3*fc, fontsize=21*fc)
    compass(annotate, angle=tot_rot, ax=None, length=8, textscale=1.6, fontsize=21*fc, \
            color='white', labeleast=True, linewidth=3*fc)

    plt.text(0.01, 0.985, panel, color='black',
             ha='left', va='top',size=18, transform=ax.transAxes)
    
    if save=='singlesave':
        ax.set_autoscale_on(False)
        plt.tight_layout()
        plt.savefig(outtitle, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    elif save=='show':
        ax.set_autoscale_on(False)
        plt.tight_layout()
        plt.show()


def doall(disk='blah'):
    pic(band='full', save='singlesave', image='data', disk=disk, panel='(a)')
    pic(band='j', save='singlesave', image='data', disk=disk, panel='(b)')
    pic(band='h', save='singlesave', image='data', disk=disk, panel='(c)')
    pic(band='k', save='singlesave', image='data', disk=disk, panel='(d)')

    pic(band='full', save='singlesave', image='snr', disk=disk, panel='(e)')
    pic(band='j', save='singlesave', image='snr', disk=disk, panel='(f)')
    pic(band='h', save='singlesave', image='snr', disk=disk, panel='(g)')
    pic(band='k', save='singlesave', image='snr', disk=disk, panel='(h)')
    

def makefigure2():
    
    images = map(Image.open, ['figs/pretty/hip79977_collapsed.png',
                              'figs/pretty/hip79977_j.png',
                              'figs/pretty/hip79977_h.png',
                              'figs/pretty/hip79977_k.png',
                              'figs/pretty/hip79977_collapsed_snr.png',
                              'figs/pretty/hip79977_j_snr.png',
                              'figs/pretty/hip79977_h_snr.png',
                              'figs/pretty/hip79977_k_snr.png'])

    x = np.shape(images[4])[1]
    y = np.shape(images[4])[0]
    
    new_im = Image.new('RGB', (x*4, y*2), color='white')

    new_im.paste(images[0], (x*0, 0))
    new_im.paste(images[1], (x*1 , 0))
    new_im.paste(images[2], (x*2 , 0))
    new_im.paste(images[3], (x*3 , 0))
    new_im.paste(images[4], (x*0 , y))
    new_im.paste(images[5], (x*1, y))
    new_im.paste(images[6], (x*2, y))
    new_im.paste(images[7], (x*3, y))

    new_im.save('figs/diskbandssnr.png')


def makefigure(save=False):
    #presently results in the lower row of figures being smaller...
    #...this is fixable but I'm not sure how. Threfore I made
    #makefigure2
    
    plt.figure(figsize=(14, 7), dpi=100)

    #gs1 = gridspec.GridSpec(3, 3)
    #gs1.update(left=0.05, right=0.48, wspace=0.05)

    plt.subplot(2,4,1)
    pic(band='full', save='multi', image='data', disk='pretty')
    plt.subplot(2,4,2)
    pic(band='j', save='multi', image='data', disk='pretty')
    plt.subplot(2,4,3)
    pic(band='h', save='multi', image='data', disk='pretty')
    plt.subplot(2,4,4)
    pic(band='k', save='multi', image='data', disk='pretty')

    plt.subplot(2,4,5)
    pic(band='full', save='multi', image='snr', disk='pretty')
    plt.subplot(2,4,6)
    pic(band='j', save='multi', image='snr', disk='pretty')
    plt.subplot(2,4,7)
    pic(band='h', save='multi', image='snr', disk='pretty')
    plt.subplot(2,4,8)
    pic(band='k', save='multi', image='snr', disk='pretty')

    plt.subplots_adjust(left=0.05, bottom=0.04, right=0.98, top=0.96, hspace=0.13, wspace=0.13)
    
    if save:
        outtitle='figs/Combined_Plots.png'
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


        

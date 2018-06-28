import matplotlib.pyplot as plt
import numpy as np
import pyfits
import pdb
from matplotlib.colors import LogNorm
#import sys
from PIL import Image, ImageDraw, ImageFont

dir = '20180627/'
#'figs/'

realdisk_fn = 'settled2_npca=2,drsub=6,nfwhm=1,rmin=5,rmax=50,meansub,meanadd_collapsed.fits'

syndisk_fn ='egrater_0.650000,3.00000,5,-2.00000,112.400,0,0,112.400,84.6000,60,1,0.fits'
#'egrater_0.600000,1.00000,4,-3.50000,112.400,0,0,112.400,84.6000,70,2,0.fits'

syndisk_psfsub_fn = 'egrater_0.650000,3.00000,5,-2.00000,112.400,0,0,112.400,84.6000,60,1,0_psfsubcol.fits'
#'egrater_0.600000,1.00000,4,-3.50000,112.400,0,0,112.400,84.6000,70,2,0_psfsubcol.fits'

tot_rot = 0#-125.22#112.4

def syndisk(save=False):
    syndisk = pyfits.getdata(dir+syndisk_fn)

    #rotate image by 90 degrees so N is up
    #syndisk = np.rot90(syndisk, k=3)
    #tot_rot += 90
    
    pixscal=4.5556e-6 * 60*60 #arcsec/pixel

    syndisk += 1e-15 #crop and make all values nonzero
    
    plt.figure(figsize=(7.5, 4), dpi=100)
    plt.imshow(syndisk, interpolation='none', vmin=1e-10, vmax=1e-6, norm=LogNorm())
    plt.colorbar(shrink=0.9)
    plt.title("Synthetic Disk", size=12)
    plt.text(100, 50, \
             r'$g=$'+syndisk_fn[8:12] + ', '+ \
             r'$r_0=$'+syndisk_fn[64:66] + ', '+\
             r'$e=$'+syndisk_fn[-6:-5] + ', '+\
             r'$\beta=$'+syndisk_fn[67:68] + ', '+\
             r'$\xi=$'+syndisk_fn[17:20] + ', '+\
             r'$\alpha_{in}=$'+syndisk_fn[25:26] + ', '+\
             r'$\alpha_{out}=$'+syndisk_fn[27:31], \
             horizontalalignment='center', \
             color='white')
    #plt.xlim(, 146)
    plt.ylim(40, 160)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    #plt.subplots_adjust(bottom=0.0, top=1)
    
    compass([30,130], angle=tot_rot, ax=None, length=8, textscale=2, fontsize=14, \
            color='white', labeleast=True, linewidth=2)
    scalebar([70, 140], distance=66, pixelscale=pixscal, linewidth=2, fontsize=14)

    plt.text(5, 155, '(a)', color='black', bbox=dict(facecolor='white'),
             ha='left', va='top',size=10)
    
    outtitle='figs/syndisk.png'
    
    if save:
        plt.savefig(outtitle, dpi=150)#, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()
        

def psfsub_syndisk(save=False):
    syndisk_psfsub = pyfits.getdata(dir+syndisk_psfsub_fn)
    #syndisk_psfsub[syndisk_psfsub < 0.1] = 0.1 #necessary for log scaling
    
    plt.figure(figsize=(7.5, 4), dpi=100)
    #plt.imshow(syndisk_psfsub, interpolation='none', vmin=0.1, vmax=10, norm=LogNorm())
    plt.imshow(syndisk_psfsub, interpolation='none', vmin=-1, vmax=6)#, norm=LogNorm())
    plt.colorbar()
    plt.title("PSF-Subtracted Synthetic Disk")
    plt.xlim(54, 146)
    plt.ylim(72, 128)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    compass([75,110], angle=tot_rot, ax=None, length=6, textscale=1.6, fontsize=18, \
            color='white', labeleast=True, linewidth=3)

    plt.text(plt.xlim()[0]+2.5, plt.ylim()[1]-2.5,
             '(b)', color='black', bbox=dict(facecolor='white'),
             ha='left', va='top', size=14)
    
    outtitle='figs/syndisk_psfsub.png'
    
    if save:
        plt.savefig(outtitle, dpi=150)#, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()
        
    
def realdisk(save=False):
    realdisk = pyfits.getdata(dir+realdisk_fn)
    
    plt.figure(figsize=(7.5, 4), dpi=100)
    plt.imshow(realdisk, interpolation='none', vmin=-1, vmax=6)#, norm=LogNorm())
    plt.colorbar()
    plt.title("Actual Disk Data")
    plt.xlim(54, 146)
    plt.ylim(72, 128)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    compass([75,110], angle=tot_rot, ax=None, length=6, textscale=1.6, fontsize=18, \
            color='white', labeleast=True, linewidth=3)
    

    plt.text(plt.xlim()[0]+2.5, plt.ylim()[1]-2.5,
             '(c)', color='black', bbox=dict(facecolor='white'),
             ha='left', va='top', size=14)

    outtitle='figs/realdisk.png'
    
    if save:
        plt.savefig(outtitle, dpi=150)#, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()

        
def difference(save=False):
    realdisk = pyfits.getdata(dir+realdisk_fn)
    syndisk_psfsub = pyfits.getdata(dir+syndisk_psfsub_fn)
    
    plt.figure(figsize=(7.5, 4), dpi=100)
    plt.imshow(realdisk - syndisk_psfsub, \
               interpolation='none', vmin=-1, vmax=6)#, norm=LogNorm())
    plt.colorbar()
    plt.title("(Actual Disk) - (PSF-Subtracted Synthetic Disk)")
    plt.xlim(54, 146)
    plt.ylim(72, 128)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    compass([75,110], angle=tot_rot, ax=None, length=6, textscale=1.6, fontsize=18, \
            color='white', labeleast=True, linewidth=3)
    

    plt.text(plt.xlim()[0]+2.5, plt.ylim()[1]-2.5,
             '(d)', color='black', bbox=dict(facecolor='white'),
             ha='left', va='top', size=14)

    outtitle='figs/realdisk-syndisk.png'
    
    if save:
        plt.savefig(outtitle, dpi=150)#, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()


def doall():
    syndisk(save=True)
    psfsub_syndisk(save=True)
    realdisk(save=True)
    difference(save=True)

    

def combine():
    images = map(Image.open, ['figs/syndisk.png',
                              'figs/syndisk_psfsub.png',
                              'figs/realdisk.png',
                              'figs/realdisk-syndisk.png'])
    
    new_im = Image.new('RGB', (1125, 2400), color='white')

    sf = 0.075 #percent to crop from each edge
    images[0] = images[0].crop(np.round((np.shape(images[0])[1]*sf,
                                         np.shape(images[0])[0]*sf,
                                         np.shape(images[0])[1]*(1.-sf),
                                         np.shape(images[0])[0]*(1.-sf))).astype(int))
    #basewidth = 1125
    #img = Image.open('somepic.jpg')
    #wpercent = (basewidth/float(images[0].size[0]))
    #hsize = int((float(images[0].size[1])*float(wpercent)))
    images[0] = images[0].resize([1125, 600], Image.ANTIALIAS)

    #images[0] = images[0][50:550 , 100:1025, :]
    #images[0].thumbnail(np.round(np.array(np.shape(images[0])[:2])*2.5).astype(int), Image.ANTIALIAS)

    new_im.paste(images[0], (10,0))
    new_im.paste(images[1], (0,600))
    new_im.paste(images[2], (0,1200))
    new_im.paste(images[3], (0,1800))

    #label the panes
    #font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    #draw = ImageDraw.Draw(new_im)
    #draw.text((10, 10),"Sample Text", font=font, fill=(255,255,255,255))#(255,255,255))
    
    new_im.save('figs/syndisk_combined.png')

    #pdb.set_trace()


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

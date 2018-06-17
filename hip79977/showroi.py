import matplotlib.pyplot as plt
import numpy as np
import pyfits
import pdb
from matplotlib.colors import LogNorm

#plots the region of interest on the wavelength-collapsed image
#also prints the number of resolution elements in the ROI

def main(save=False):
    
    lrmin=10
    lrmax=45
    roirange=[100,30,-22]
    center = [100,100]
    tot_rot = 0#-125.22 #rotation of disk from N. degrees. get from fits header
    
    file = 'figs/ugly/settled2_npca=2,drsub=6,nfwhm=1,rmin=5,rmax=50,meansub,meanadd_collapsed.fits'
    img = pyfits.getdata(file)
    
    #rotate image by 90 degrees so N is up
    #img = np.rot90(img, k=3)
    #tot_rot += 90
    #roirange[2] = tot_rot - roirange[2]
    roirange[2] *= -1
    print roirange[2]

    #define vectors pointing toward the corners of the box
    #The box is horizontal and centered at the origin
    v1 = [0.5 * roirange[0] , 0.5 * roirange[1] ]
    v2 = [-0.5 *roirange[0] , 0.5 * roirange[1] ]
    v3 = [-0.5 *roirange[0] , -0.5 * roirange[1]]
    v4 = [0.5 * roirange[0] , -0.5 * roirange[1]]

    #define rotation matrix
    R = [[ np.cos(np.deg2rad(roirange[2])) , -1*np.sin(np.deg2rad(roirange[2])) ] ,
         [ np.sin(np.deg2rad(roirange[2])) ,    np.cos(np.deg2rad(roirange[2])) ] ]

    #Find coordinates of the corners of the box after rotation
    x1,y1 = np.dot(R, v1)
    x2,y2 = np.dot(R, v2)
    x3,y3 = np.dot(R, v3)
    x4,y4 = np.dot(R, v4)

    #Find a and b in y=ax+b for top and bottom of the box
    a = np.tan(np.deg2rad(roirange[2]))
    b_t = y1 - x1 * np.tan(np.deg2rad(roirange[2]))
    b_b = y4 - x4 * np.tan(np.deg2rad(roirange[2]))

    r = lrmax
    #Find new X and Y of the corners of the box where it intersects the circle
    # defined by lrmax. I'm not showing the derivation of this, sorry.
    nx1 = (-1*a*b_t + np.sqrt(-1 * b_t**2 + r**2 + a**2 * r**2) ) / (1+a**2)
    nx2 = (-1*a*b_t - np.sqrt(-1 * b_t**2 + r**2 + a**2 * r**2) ) / (1+a**2)
    nx3 = (-1*a*b_b - np.sqrt(-1 * b_b**2 + r**2 + a**2 * r**2) ) / (1+a**2)
    nx4 = (-1*a*b_b + np.sqrt(-1 * b_b**2 + r**2 + a**2 * r**2) ) / (1+a**2)
    ny1 = (b_t - np.sqrt(a**2 * r**2 - a**2 * b_t**2 + a**4 * r**2) ) / (1+a**2)
    ny2 = (b_t + np.sqrt(a**2 * r**2 - a**2 * b_t**2 + a**4 * r**2) ) / (1+a**2)
    ny3 = (b_b + np.sqrt(a**2 * r**2 - a**2 * b_b**2 + a**4 * r**2) ) / (1+a**2)
    ny4 = (b_b - np.sqrt(a**2 * r**2 - a**2 * b_b**2 + a**4 * r**2) ) / (1+a**2)

    #Calculate theta values where lrmax circle intersects rectangle
    theta1 = np.arctan2(ny1 , nx2)
    theta2 = np.arctan2(ny2 , nx1)
    theta3 = np.arctan2(ny3 , nx4)
    theta4 = np.arctan2(ny4 , nx3)

    print "Theta 1, 2, 3, 4:", np.rad2deg([theta1, theta2, theta3, theta4])

    mymin=0#-10000
    mymax=6#60000
    title="Region of Interest for "+r'$\chi^2$'+"Calculation"
    outtitle = 'figs/roi.png'

    #create map showing radius^2 at each point
    #scaling = np.zeros(np.shape(img))
    #for x in range(np.shape(scaling)[1]):
    #    for y in range(np.shape(scaling)[0]):
    #        scaling[y, x] = (x-float(center[0]))**2+(y-float(center[1]))**2
            
    #Actually apply radial scaling
    #img *= scaling
    
    pixscal=4.5556e-6 * 60*60 #arcsec/pixel
           
    plt.imshow(img, interpolation='none', vmin=mymin, vmax=mymax)#, norm=LogNorm())
    plt.title(title, fontsize=21)
    plt.xlim(53,147)
    plt.ylim(53,147)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.tight_layout()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    compass([80,120], angle=tot_rot, ax=None, length=8, textscale=1.6, fontsize=21, \
            color='white', labeleast=True, linewidth=3) #add compass to image
    scalebar([95, 133], distance=66, pixelscale=pixscal, linewidth=3, fontsize=21)

    #plot center marker
    plt.plot(center[0], center[1], 'k+', markersize=25, markeredgewidth=3)

    #plot two sides of inner rectangle
    plt.plot(np.array([nx1, nx2])+center, np.array([ny2, ny1])+center, \
             'y-', linewidth=2)
    plt.plot(np.array([nx4, nx3])+center, np.array([ny3, ny4])+center, \
             'y-', linewidth=2)

    #plot the arcs defined by lrmax
    plt.plot(lrmax * np.cos(np.linspace(theta2, theta3, 50)) + center[0] , \
             lrmax * np.sin(np.linspace(theta2, theta3, 50)) + center[1] , \
             'y-', linewidth=2)
    plt.plot(lrmax * np.cos(np.linspace(theta1, theta4, 50)) + center[0] , \
             lrmax * np.sin(np.linspace(theta1, theta4, 50)) + center[1] , \
             'y-', linewidth=2)

    #plot the lrmin circle
    theta = np.arange(0, 2*np.pi, 0.01)
    plt.plot(lrmin * np.cos(theta) + center[0], \
             lrmin * np.sin(theta) + center[1], \
             'y-', linewidth=2)

    if save:
        plt.savefig(outtitle, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()


    #Now calculate the number of resolution elements in the ROI
    lambdaa = 1.63e-6 #middle wavelength
    D = 7.9 #diameter of primary
    pix_pitch = 4.5555e-6 / 360 * 2 * np.pi #radians per pixel
    pix_per_res = np.pi * (lambdaa / D / 2)**2 / pix_pitch**2
    print
    print "Pixels per resolution element:", pix_per_res
    x = np.sqrt( (nx1 - nx2)**2 + (ny1 - ny2)**2) #width of new box
    y = np.sqrt( (nx1 - nx4)**2 + (ny1 - ny4)**2) #height of new box
    a_whole = x * y / 2 + 2 * lrmax**2 * np.arctan(y/x) #area of region in pixels
    a_whole -= (np.pi * lrmin**2) #subtract inner region
    print "Pixels in region:", a_whole
    print "Resolution elements in region:", a_whole/pix_per_res
    print



    
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

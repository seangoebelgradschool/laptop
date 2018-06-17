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

    #fit to data
    jec = np.polyfit(jer[np.isfinite(je)], je[np.isfinite(je)], 1, w=2.5**(-1*jeu[np.isfinite(je)]))
    jef = np.poly1d(jec)
    print "J band east exponential falloff:", jec[0]
    jwc = np.polyfit(jwr[np.isfinite(jw)], jw[np.isfinite(jw)], 1, w=2.5**(-1*jwu[np.isfinite(jw)]))
    jwf = np.poly1d(jwc)
    print "J band west exponential falloff:", jwc[0]

    hec = np.polyfit(her[np.isfinite(he)], he[np.isfinite(he)], 1, w=2.5**(-1*heu[np.isfinite(he)]))
    hef = np.poly1d(hec)
    print "h band east exponential falloff:", hec[0]
    hwc = np.polyfit(hwr[np.isfinite(hw)], hw[np.isfinite(hw)], 1, w=2.5**(-1*hwu[np.isfinite(hw)]))
    hwf = np.poly1d(hwc)
    print "H band west exponential falloff:", hwc[0]

    kec = np.polyfit(ker[np.isfinite(ke)], ke[np.isfinite(ke)], 1, w=2.5**(-1*keu[np.isfinite(ke)]))
    kef = np.poly1d(kec)
    print "K band east exponential falloff:", kec[0]
    kwc = np.polyfit(kwr[np.isfinite(kw)], kw[np.isfinite(kw)], 1, w=2.5**(-1*kwu[np.isfinite(kw)]))
    kwf = np.poly1d(kwc)
    print "K band west exponential falloff:", kwc[0]

    slopes = np.array([jec[0], jwc[0], hec[0], hwc[0], kec[0], kwc[0]])
    print "average slope", np.mean(slopes)
    print "slope stddev", np.std(slopes, ddof=1)
            
                     
    #pdb.set_trace()
    
    plt.figure(figsize=(7.5, 10), dpi=100)

    #J BAND
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.errorbar(jer, je, yerr=jeu, fmt='bo', label='East')
    plt.errorbar(jwr, jw, yerr=jwu, fmt='ro', label='West')
    xp = np.linspace(0,1.2, 100)
    plt.plot(xp, jef(xp), 'b-', label='fit=-'+str(jec[0])[:4])
    plt.plot(xp, jwf(xp), 'r-', label='fit=-'+str(jwc[0])[:4])
    plt.title(r'$J$'+ ' Band Surface Brightness of Disk Spine', fontsize=20)
    plt.ylabel('Magnitude per arcsec'+r'$^2$', fontsize=16)
    plt.xlabel('R [arcsec]', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.set_autoscale_on(False)
    plt.xlim(0.1, 1.1)
    plt.ylim(17, 8)

    #H BAND
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.errorbar(her, he, yerr=heu, fmt='bo', label='East')
    plt.errorbar(hwr, hw, yerr=hwu, fmt='ro', label='West')
    #xp = np.linspace(0,1.2, 100)
    plt.plot(xp, hef(xp), 'b-', label='fit=-'+str(hec[0])[:4])
    plt.plot(xp, hwf(xp), 'r-', label='fit=-'+str(hwc[0])[:4])

    plt.title(r'$H$'+ ' Band Surface Brightness of Disk Spine', fontsize=20)
    plt.ylabel('Magnitude per arcsec'+r'$^2$', fontsize=16)
    plt.xlabel('R [arcsec]', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.set_autoscale_on(False)
    plt.xlim(0.1, 1.1)
    plt.ylim(17, 8)

    #K BAND
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.errorbar(ker, ke, yerr=keu, fmt='bo', label='East')
    plt.errorbar(kwr, kw, yerr=kwu, fmt='ro', label='West')
    xp = np.linspace(0,1.2, 100)
    plt.plot(xp, kef(xp), 'b-', label='fit=-'+str(kec[0])[:4])
    plt.plot(xp, kwf(xp), 'r-', label='fit=-'+str(kwc[0])[:4])

    plt.title(r'$K_p$'+ ' Band Surface Brightness of Disk Spine', fontsize=20)
    plt.ylabel('Magnitude per arcsec'+r'$^2$', fontsize=16)
    plt.xlabel('R [arcsec]', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.set_autoscale_on(False)
    plt.xlim(0.1, 1.1)
    plt.ylim(17, 8)

    plt.subplots_adjust(left=0.09, bottom=0.07, right=0.97, top=0.96, hspace=0.40)
    

    if save:
        outtitle = 'figs/surfacebrightness2.png'
        plt.savefig(outtitle, bbox_inches='tight', dpi=200)
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()

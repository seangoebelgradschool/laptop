import matplotlib.pyplot as plt
import numpy as np
import pdb
import matplotlib.ticker as mtick

power_0 = 1.96e-9 #W/m^2 in H band for 0 mag star. Actually is a flux, not a power.
h = 6.626e-34 #J s 
nu = 3e8 / 1.64e-6 #c/lambda
A = np.pi * ((8.2)/2)**2
throughput = 0.1 #absorption by telescope and instrument

photons_per_speckle = 10 #how many photons do we want in a speckle?

E = h * nu #joules per photon


def calc():
    framerate = (np.linspace(10,100,91))**2

    mag_1 = 1 #what mag are we looking at?
    mag_4 = 4 #what mag are we looking at?
    mag_7 = 7 #what mag are we looking at?
    
    photon_flux_1 = power_0 * (2.512**(-1.*mag_1)) / (h*nu) * A
    contrast_1 = (throughput * photon_flux_1 * framerate**-1 /
                photons_per_speckle)**(-1)

    photon_flux_4 = power_0 * (2.512**(-1.*mag_4)) / (h*nu) * A
    contrast_4 = (throughput * photon_flux_4 * framerate**-1 /
                photons_per_speckle)**(-1)

    photon_flux_7 = power_0 * (2.512**(-1.*mag_7)) / (h*nu) * A
    contrast_7 = (throughput * photon_flux_7 * framerate**-1 /
                photons_per_speckle)**(-1)

    plt.loglog(framerate, contrast_1,
               framerate, contrast_4,
               framerate, contrast_7)
    plt.gca().invert_yaxis()
    #plt.xscale('log')
    plt.title("Framerate vs. Contrast for "+ str(photons_per_speckle) +
              " Photons per Speckle")
    plt.xlabel('Loop Frame Rate (Hz)')
    plt.ylabel('Speckle Contrast')
    plt.legend(('Stellar mag H=1', 'Stellar mag H=4', 'Stellar mag H=7'))
    plt.show()

def pt2():
    const = throughput * power_0 * A / (h * nu * photons_per_speckle)

    mags = np.arange(0,8,0.1)
    contrast_solns = np.array([])
    for H in mags:
        con_arr = np.sort(np.reshape(np.array([np.arange(1,10, 0.01)*1e-2 ,
                                               np.arange(1,10, 0.01)*1e-3 , 
                                               np.arange(1,10, 0.01)*1e-4 ,
                                               np.arange(1,10, 0.01)*1e-5 ,
                                               np.arange(1,10, 0.01)*1e-6 ,
                                               np.arange(1,10, 0.01)*1e-7 ,
                                               np.arange(1,10, 0.01)*1e-8 ]), 9*7*100))
        eqnpt1 = con_arr**3
        eqnpt2 = 0.3936 * 2.512**(2.*H) / const**2 + \
                 0.0232 * 2.512**H * con_arr / const

        loc = np.where(abs(np.log(eqnpt1) - np.log(eqnpt2)) ==
                       np.min(abs(np.log(eqnpt1) - np.log(eqnpt2))))
        contrast_solns = np.append(contrast_solns, con_arr[loc])

        if 0: #check that solution is solved right
            print con_arr[loc]
            
            plt.loglog(con_arr, eqnpt1)
            plt.loglog(con_arr, eqnpt2)
            plt.axvline(con_arr[loc])
            plt.show()

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    plt.plot(mags, contrast_solns)
    plt.yscale('log')
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    plt.title("Flux-limited Achievable Raw Contrast for SCExAO")
    plt.xlabel('H band Magnitude')
    plt.ylabel('Theoretically Achievable Raw Contrast')
    plt.show()

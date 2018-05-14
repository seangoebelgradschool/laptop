import matplotlib.pyplot as plt
import numpy as np
import pyfits
import os

#figure out a mask that fits the disk

def main():
    dir = 'snrmaps/'
    files = os.listdir(dir)

    img = np.sum(pyfits.getdata(dir+files[1]), 0)

    theta = np.linspace(0, 2*np.pi, 1000)
    lamb = np.deg2rad(78)
    a = 65*2 #pixels
    b = 12 #pixels
    xshift = 95 #pixels
    yshift = 100 #pixels

    if 0:
        x = a * np.cos(theta)
        y = b * np.sin(theta)

    else:
        x = np.array([np.linspace(-0.5 * a, 0.5*a, 100) , \
                      np.zeros(100) + 0.5 * a , \
                      np.linspace(-0.5 * a, 0.5*a, 100) , \
                      np.zeros(100) - 0.5 * a ])
        y = np.array([np.zeros(100) + 0.5 * b , \
                      np.linspace(-0.5 * b, 0.5*b, 100) , \
                      np.zeros(100) - 0.5 * b , \
                      np.linspace(-0.5 * b, 0.5 * b, 100) ])
        
    xp = x * np.cos(lamb) - y * np.sin(lamb) + xshift
    yp = x * np.sin(lamb) + y * np.cos(lamb) + yshift
    
    plt.imshow(img, interpolation='none', vmin=-20, vmax=20)
    plt.colorbar()
    plt.plot(xp, yp, '*')
    plt.show()

import matplotlib.pyplt as plt
import numpy as np

t=2000 #integration time
rn = 
d = 0.011 #dark current /s

#Photons from 1 arcsec^2 reaching detector per sec in J band
sky_bg = 1e4 * (30. / 2.)^2 * np.pi * 0.8 * 0.3

sky_per_pix = sky_bg * 0.01^2 * 

import matplotlib.pyplot as plt
import numpy as np
import pdb

im = np.zeros((3,3))
im[1,1] = 2
ft1 = np.fft.fft2(im)
print im
print ft1
print

im = np.zeros((3,3))
im[1,2] = 2
ft2 = np.fft.fft2(im)
print im
print ft2
print

print ft2-ft1
print

inv = np.fft.ifft2(ft2-ft1)
print inv
print

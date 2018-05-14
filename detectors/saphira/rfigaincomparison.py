import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.stats import mode

#gain ~ 30, bias = -12.5V
highgain = pyfits.getdata('images/saphira_08:02:52.773012749_cleaned.fits') 

#gain ~1, 1V common
lowgain = pyfits.getdata('images/saphira_07:03:37.097619389_cleaned.fits') 

print "Size low gain, high gain", np.shape(lowgain), np.shape(highgain)

mymin = np.sort(highgain[0].flatten())[np.size(highgain[0])*.01]
mymax = np.sort(highgain[0].flatten())[np.size(highgain[0])*.99]

for i in range(0):#100):
    plt.figure(num=1, figsize=(10, 4), dpi=100) 

    plt.subplot(121)
    plt.imshow(highgain[i], interpolation='none', vmin=mymin, vmax=mymax)
    plt.title('High Gain')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(lowgain[i], interpolation='none', vmin=mymin, vmax=mymax)
    plt.title("Low Gain")

    plt.show()

plt.figure(num=1, figsize=(7,10), dpi=100)

plt.subplot(211)
plt.hist(lowgain[:].flatten(), bins=50, range=[-50, 50], normed=True, log=True)#, alpha=0.5)
plt.title("2.5V Bias, Gain~1")
plt.xlabel("ADU")

plt.subplot(212)
plt.hist(highgain[:].flatten(), bins=50, range=[-50, 50], normed=True, log=True)#, alpha=0.5)
plt.title("12.5V Bias, Gain~29")
plt.xlabel("ADU")

plt.show()

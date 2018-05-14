import numpy as np
import pyfits
import matplotlib.pyplot as plt
import pdb

def gain():
    im_names=['prv3.38forreal-1-0.fits',
              'prv3.48-2-0.fits',
              'prv3.58serious-2-0.fits',
              'prv3.67forreal-1-0.fits']

    v = np.array([3.38, 3.48, 3.58, 3.67])

    increments=np.arange(10)*32
    storage = np.array([]) #store the volt gains
    #x = np.append(np.arange(50), np.arange(40)+280)

    for j in range(32):
        adus=np.zeros(4)
        for i in range(4):
            img = pyfits.getdata('images/'+im_names[i])
            #select masked off, unilluminated pixels
            selection = (increments+j)[np.where((increments+j < 50) |
                                                (increments+j >280))]
            adus[i] = np.median(img[10: , :, selection])
            
        plt.plot(v, adus, 'ro')
        coeffs=np.polyfit(v, adus, 1)
        p = np.poly1d(coeffs)
        plt.plot(v, p(v), 'b-')
        #plt.title(str(j)+' ' + str(coeffs))
        #plt.show()

        storage = np.append(storage, coeffs[0]**-1*1e6)
        print j, storage[j]
        

    print "median:", np.median(storage), "uV/ADU"
        
    plt.title('Voltage Gain')
    plt.xlabel('PRV (Volts)')
    plt.ylabel('ADU')
    plt.ylim([21e3, 36e3])
    plt.show()

import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import pdb

#Works when placed in the /home/pizzabox/Desktop/SAPHIRA.26FEB2018/LABTEST-SAPHIRA/pbserver.lt directory. Was used to calculate volt gains for pb lab tests w and without cryo preamp. See 3/3/2018 email thread for more information.

def main():
    if 0:#1: #volt gain with preamp
        volts = [3.562, 3.572, 3.581, 3.591, 3.601, 3.612, 3.621]
        n_frames = 3996
        n_exclude = 2000
        date = '180302'
        mytitle="Volt Gain with Mk15 SAPHIRA M09105-27 with Cryo Preamp"
        sigfigs = 4
        
    else:#volt gain with Henriksen mount
        volts = [3.5793, 3.5890, 3.5988, 3.6088, 3.6182, 3.6292, 3.6389]
        n_frames = 9993
        n_exclude = 6000
        date = '180313'
        mytitle="Volt Gain with Mk15 SAPHIRA M09105-27 in JK Henriksen Mount"
        sigfigs = 5
        
    reset_freq = 200 #how often is detector reset?
    avg_adus = np.zeros(len(volts))
    
    for j in range(len(volts)):
        voltage = str(volts[j])
        while len(voltage) < sigfigs+1: voltage = voltage+'0' #python drops trailing 0s
        
        meds = np.zeros(n_frames)
        print "Working on dataset " + str(j+1) + " of " + str(len(volts)) + "."
        for i in range(len(meds)):
            if i>n_exclude: #exclude first 1000 frames
                if (i%reset_freq > 75): #not in first 75 frames after reset
                    filenum = str(i)
                    while len(filenum) <4: filenum = '0'+filenum
                    rampno='02'
                    meds[i] = np.median(pyfits.getdata('/home/pizzabox/Desktop/SAPHIRA.26FEB2018/LABTEST-SAPHIRA/pbserver.lt/'+date+'_Voltgain_PRV_'+voltage+'V-'+rampno+'-'+filenum+'.fits'))

        meds = meds[np.where(meds != 0)] #remove empty elements

        coeffs=np.polyfit(range(len(meds)), meds, 1)
        p = np.poly1d(coeffs)
    
        avg_adus[j] = coeffs[1]
        #avg_adus[j] = np.mean(meds)

        if j==1:
            plt.plot(meds, 'o')
            plt.plot([0, len(meds)], [np.mean(meds), np.mean(meds)], \
                     label='mean=' + str(np.mean(meds))[:7])
            plt.plot(range(len(meds)), p(range(len(meds))), \
                     label='ADUs = '+str(coeffs[0])[:7]+'x + ' + str(coeffs[1])[:7] )
            plt.title(str(voltage)+ 'V')
            plt.legend(loc=1)
            plt.show()
        #pdb.set_trace()
            
    coeffs=np.polyfit(volts, avg_adus, 1)
    p = np.poly1d(coeffs)

    print "Volt gain is: ", 1.e6 / coeffs[0], 'uV/ADU'
    
    plt.plot(volts, avg_adus, 'o')
    plt.plot(volts, p(volts), '-', label=str(1.e6 / coeffs[0])[1:6]+' uV/ADU')
    plt.title(mytitle)
    plt.xlabel('PRV [V]')
    plt.ylabel('Mean ADUs')
    plt.legend(loc=1)
    plt.show()

    np.savez(mytitle, mytitle, volts, avg_adus, coeffs, p(volts))

    pdb.set_trace()

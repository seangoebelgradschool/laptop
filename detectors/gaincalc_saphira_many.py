#/usr/bin/env python
#Updated 2/10/14

#Calculates gain by looking at the spatial variance and median flux of crops
# from illuminated data sets. Also calculates the read noise. Can be called
# from the command line with
# "python gaincalc.py"

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

#crosstalk = 0.0 #dunno
dir = 'saphira/'
filename = '150317_151633.fits' #w/ 12 crops, mean/median =2.1 e-/adu, 29 ADU read noise
img = pyfits.getdata(dir+filename)

frames = np.arange(3, 40, 1) #frame numbers to loop over
#center = ((f9.shape)[1])/2 #pixel number of center of frame
#yshifts = np.zeros(n_crops) #vertical shifts of the crops
noisearr = np.zeros((len(frames) , 0)) #stores measured variances
fluxarr = np.zeros((len(frames) , 0)) #stores measured flux values
fits = np.zeros((3, 0)) #quadratic fitting terms for each of the crops
yshifts = []
xshifts = []
masks = []

#print "Crosstalk correction factor is", (str(crosstalk*100))[0:4], '%.'
#correction = (5.*crosstalk + 1.) / (crosstalk + 1.)
#print
print "Please decide if the following crops are reasonably uniformly illuminated and "\
 "free of major defects. Some bad pixels are alright."

#Displays crops for user to decide if they're good. If user says the crop is good,
# its location is saved and the code shifts 64 pixels to the side. If the user
# says the crop is bad, it shifts 70 pixels vertically and tries again. This way,
# the crops are spread out across the readout columns.
for x in range(32,96,32):
    for y in range(32, 256, 32):#np.shape(img)[1], 32):
        answer = 'n'

        crop = img[2, y:y+32, x:x+32] - img[99, y:y+32 , x:x+32]
        #crop = img[2, 0:25 , 0:25] - img[99, 0:25 , 0:25]
        plt.imshow(crop, interpolation='nearest', 
                   vmin=np.median(crop)-5.*np.sqrt(np.median(crop)),
                   vmax=np.median(crop)+5.*np.sqrt(np.median(crop)) ) #show the frame
        #plt.title('Finding Crop '+str(i+1))
        plt.colorbar()
        plt.show()

        #ask user if the crop is acceptable
        answer = raw_input('Is the crop clean (y/n)? ')
        print
        if answer == 'y':
            xshifts = np.append(xshifts, x)
            yshifts = np.append(yshifts, y)
            noisearr = np.append(noisearr, np.zeros((len(frames),1)), 1)
            fluxarr = np.append(fluxarr, np.zeros((len(frames),1)), 1)
            fits = np.append(fits, np.zeros((3,1)),1)

    #Select central 60% of pixels, which hopefully don't include outlier populations
            mask = np.where( (crop >  (np.sort(crop.flatten()))[crop.size * 0.2]) & 
                             (crop <  (np.sort(crop.flatten()))[crop.size * 0.8]) )

            myrange = [np.median(crop)-9.*np.std(crop[mask]) , 
                       np.median(crop)+9.*np.std(crop[mask])]
            junk = plt.hist(crop.flatten(), bins=50, range=myrange)
            mode = (junk[1])[np.where(junk[0] == np.max(junk[0]))] #approximately
            plt.clf()

    #calculate mask with bad pixels in place
            mask = np.where( (crop > (mode - 3.*np.std(crop[mask], ddof=1))) &
                             (crop < (mode + 3.*np.std(crop[mask], ddof=1))) )

    #mask = np.where(maskcrop == 1)

            print (1.-float(crop[mask].size)/float(crop.size))*100., \
                "% of pixels were rejected as bad."

    #check masking?
            if 1: 
                junk = plt.hist(crop.flatten(), bins=50, range=myrange)
                plt.title("Pixel Brightness Histogram")#, CDS "+str(frames[i])+'-03')
                plt.xlabel('ADU')
                plt.ylabel('N_Occurences')
                plt.vlines( (np.min(crop[mask]) , np.max(crop[mask])) , 
                            0.1, 1.2*np.max(junk[0]), colors='r')
                plt.vlines( (mode) , 0.1, 1.2*np.max(junk[0]), colors='g')
                plt.show()
        
            for j in range(len(frames)): #cds frame delta
                crop = img[2, y:y+32, x:x+32] - img[frames[j], y:y+32 , x:x+32]

                noisearr[j, np.shape(noisearr)[1]-1] = np.var(crop[mask], ddof=1)# * correction
                fluxarr[j, np.shape(noisearr)[1]-1] = np.mean(crop[mask])

for i in range(np.shape(noisearr)[1]): #loop over n_crops
    #pdb.set_trace()
    #NOISE PLOT
    plt.plot(fluxarr[:,i], noisearr[:,i] , 'o')

    plt.title('Mean Flux vs. Spatial Variance')
    plt.xlabel('Flux [ADU]')
    plt.ylabel('Variance [ADU^2]')

    #fit a polynomial to the variance data
    coeffs = np.polyfit(fluxarr[:,i], noisearr[:,i], 2) #quadratic fit
    fits[ : , i] = coeffs #store coeffs
    fit = np.poly1d(coeffs)
    noisefit = fit(fluxarr[:,i])
    
    plt.plot(fluxarr[:,i], noisefit)
    plt.text(fluxarr[1,i],0.8*np.max(noisearr[:,i]), 'y='+str(coeffs[0])[0:6]+'x^2+'+
             str(coeffs[1])[0:4]+'x+' + str(coeffs[2])[0:4])

    plt.show()

#print np.transpose(fits)
readnoise = np.sqrt(fits[2, : ]) #convert to ADU

print "Read noise values [ADU]:", readnoise
print "Mean [ADU]:", np.mean(readnoise)
print "Median [ADU]:",  np.median(readnoise)
print "Uncertainty (sigma/root(N)) [ADU]:", np.std(readnoise, ddof=1) / np.sqrt(np.shape(noisearr)[1])
print


#NOW CALCULATE THE GAIN
gain = 1./fits[1, : ]
print "Gain [e-/ADU] values:", gain
print "Mean gain [e-/ADU]:" , np.mean(gain)
print "Median gain [e-/ADU]:", np.median(gain)
print "Gain uncertainty (sigma/root(N)) [e-/ADU]:", np.std(gain, ddof=1) / np.sqrt(np.shape(noisearr)[1])

pdb.set_trace() #Returns to command line for debugging/additional commands


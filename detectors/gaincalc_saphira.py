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
filename = '150320_162839.fits'
cols = np.arange(10)*32+2
img = pyfits.getdata(dir+filename)[:,:,cols]

n_crops = 1 #Number of crops to check, should be integer

files = np.arange(3, 100, 1) #frame numbers to loop over
#center = ((f9.shape)[1])/2 #pixel number of center of frame
#yshifts = np.zeros(n_crops) #vertical shifts of the crops
noisearr = np.zeros((len(files) , n_crops)) #stores measured variances
fluxarr = np.zeros((len(files) , n_crops)) #stores measured flux values
fits = np.zeros((3, n_crops)) #quadratic fitting terms for each of the crops

#print "Crosstalk correction factor is", (str(crosstalk*100))[0:4], '%.'
#correction = (5.*crosstalk + 1.) / (crosstalk + 1.)
#print
print "Please decide if the following crops are reasonably uniformly illuminated and "\
 "free of major defects. Some bad pixels are alright."

#Displays crops for user to decide if they're good. If user says the crop is good,
# its location is saved and the code shifts 64 pixels to the side. If the user
# says the crop is bad, it shifts 70 pixels vertically and tries again. This way,
# the crops are spread out across the readout columns.
for i in range(n_crops):
    answer = 'n'
    yshift = 0
    while (answer != 'y'):
        #frame = pyfits.getdata(dir+filename_sub + str(files[len(files)]) +
        #                       '.fits') #read in frame from 3/4 of way through ramp
        
        #crop  = ((f9-frame))[center+yshift : center+70+yshift , 
        #                   center+1+(i-n_crops/2)*64 : center-1+(i-n_crops/2)*64 +64]
        crop = img[2, 5:45 , 2:] - img[99, 5:45 , 2:]
        #crop = img[2, 0:25 , 0:25] - img[99, 0:25 , 0:25]
        plt.imshow(crop, interpolation='nearest', 
                   vmin=np.median(crop)-5.*np.sqrt(np.median(crop)),
                   vmax=np.median(crop)+5.*np.sqrt(np.median(crop)) ) #show the frame
        plt.title('Finding Crop '+str(i+1)+' of '+str(n_crops))
        plt.colorbar()
        plt.show()

        #ask user if the crop is acceptable
        answer = raw_input('Is the crop clean (y/n)? ')
        print
        #if (answer.lower() != 'y'): yshift += 70 #shift elsewhere in the column
        #if (yshift > center-71): yshift = -1.*center #don't go off the top of the image
        #yshifts[i] = yshift

    #crop = cds[center + int(yshifts[j])    : center + int(yshifts[j]) + 70 , 
    #           center+1 + (j-n_crops/2)*64 : center-1 + (j-n_crops/2)*64 + 64]
    #crop = cds[int(yshifts[j])    : int(yshifts[j]) + 70 , 
    #           center+1 + (j-n_crops/2)*64 : center-1 + (j-n_crops/2)*64 + 64]

    #maskcrop = mask_img[int(yshifts[j])    : int(yshifts[j]) + 70 , 
    #           center+1 + (j-n_crops/2)*64 : center-1 + (j-n_crops/2)*64 + 64]

    #Select central 60% of pixels, which hopefully don't include outlier populations
    mask = np.where( (crop >  (np.sort(crop.flatten()))[crop.size * 0.2]) & 
                     (crop <  (np.sort(crop.flatten()))[crop.size * 0.8]) )

    myrange = [np.median(crop)-9.*np.std(crop[mask]) , 
               np.median(crop)+9.*np.std(crop[mask])]
    junk = plt.hist(crop.flatten(), bins=50, range=myrange)
    mode = (junk[1])[np.where(junk[0] == np.max(junk[0]))] #approximately
    plt.clf()
    mode = np.median(crop[mask]) #I'M SO SORRY

    #calculate mask with bad pixels in place
    mask = np.where( (crop > (mode - 3.*np.std(crop[mask], ddof=1))) &
                     (crop < (mode + 3.*np.std(crop[mask], ddof=1))) )

    #mask = np.where(maskcrop == 1)

    print (1.-float(crop[mask].size)/float(crop.size))*100., \
        "% of pixels were rejected as bad."

    #check masking?
    if 1: 
        junk = plt.hist(crop.flatten(), bins=50, range=myrange)
        plt.title("Pixel Brightness Histogram, CDS "+str(files[i])+'-09')
        plt.xlabel('ADU')
        plt.ylabel('N_Occurences')
        plt.vlines( (np.min(crop[mask]) , np.max(crop[mask])) , 
                    0.1, 1.2*np.max(junk[0]), colors='r')
        plt.vlines( (mode) , 0.1, 1.2*np.max(junk[0]), colors='g')
        plt.show()
        
    for j in range(len(files)): #cds frame delta
        #frame = pyfits.getdata(dir+filename_sub+str(files[j])+'.fits')
        #crop  = 1.*(f9-frame)[yshift : 70+yshift , 
        #                     center+1+(i-n_crops/2)*64 : center-1+(i-n_crops/2)*64 +64]

        crop = img[2, 5:45 , 2:] - img[files[j], 5:45 , 2:]
        #crop = img[2, 0:25, 0:25] - img[files[j], 0:25 , 0:25]
   
        #crosstalk correction increases the variance but does not affect the signal
        noisearr[j, i] = np.var(crop[mask], ddof=1)# * correction
        fluxarr[j, i] = np.mean(crop[mask])
        #print "gain guess:", np.median(crop[mask]) / np.var(crop[mask])

#for i in range(n_crops): #loop over n_crops
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
print "Uncertainty (sigma/root(N)) [ADU]:", np.std(readnoise, ddof=1) / np.sqrt(n_crops)
print


#NOW CALCULATE THE GAIN
gain = 1./fits[1, : ]
print "Gain [e-/ADU] values:", gain
print "Mean gain [e-/ADU]:" , np.mean(gain)
print "Median gain [e-/ADU]:", np.median(gain)
print "Gain uncertainty (sigma/root(N)) [e-/ADU]:", np.std(gain, ddof=1) / np.sqrt(n_crops)

#pdb.set_trace() #Returns to command line for debugging/additional commands


#/usr/bin/env python
#Updated 5/23/16

#Calculates gain by looking at the spatial variance and median flux of crops
# from illuminated data sets. Also calculates the read noise. Can be called
# from the command line with
# "python gaincalc.py"
#It also corrects for crosstalk.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

#H2RG
#crosstalk = 0.005 #close enough to the 0.3 and 0.7% values
crosstalk = 0.011 #close enough to the 0.3 and 0.7% values
dir = 'h4rg/'#gaintest2/'
filename_sub = 'H4RG_R01_M01_N'
f9 = pyfits.getdata(dir+filename_sub+'49.fits')

n_crops = 10#Number of crops to check, should be integer
files = np.arange(50, 70, 1) #frame numbers to loop over

center = ((f9.shape)[1])/2 #pixel number of center of frame
noisearr = np.zeros((len(files) , n_crops)) #stores measured variances
fluxarr = np.zeros((len(files) , n_crops)) #stores measured flux values
fits = np.zeros((3, n_crops)) #quadratic fitting terms for each of the crops
yshifts = np.zeros(n_crops) #stores where the crops are located
masks = [] #stores the bad pixel masks
fluxdir = 1 #ADUs increase with flux

print "Crosstalk correction factor is", (str(crosstalk*100))[0:4], '%.'
correction = (5.*crosstalk + 1.) / (crosstalk + 1.)
print
print "Please decide if the following crops are reasonably uniformly illuminated and "\
 "free of major defects. Some bad pixels are alright."

#Displays crops for user to decide if they're good. If user says the crop is good,
# its location is saved and the code shifts 64 pixels to the side. If the user
# says the crop is bad, it shifts 70 pixels vertically and tries again. This way,
# the crops are spread out across the readout columns.
frame = pyfits.getdata(dir+filename_sub + str(max(files)) +'.fits') #read frame from end of ramp
for i in range(n_crops):
    answer = 'n'
    yshift = 0
    while (answer != 'y'):
        crop  = (fluxdir*(frame-f9))[center+yshift : center+yshift+70 , 
                                     center+1+(i-n_crops/2)*64 : center-1+(i-n_crops/2)*64 +64]
        plt.imshow(crop, interpolation='nearest', 
                   vmin=np.median(crop)-5.*np.sqrt(np.median(crop)),
                   vmax=np.median(crop)+5.*np.sqrt(np.median(crop)) ) #show the frame
        plt.title('Finding Crop '+str(i+1)+' of '+str(n_crops))
        plt.colorbar()
        plt.show()

        #ask user if the crop is acceptable
        answer = raw_input('Is the crop clean (y/n)? ')
        print
        if (answer.lower() != 'y'): yshift += 70 #shift elsewhere in the column
        if (yshift > center-71): yshift = -1.*center #don't go off the top of the image
        yshifts[i] = yshift

    #Select central 60% of pixels, which hopefully don't include outlier populations
    mask = np.where( (crop >  (np.sort(crop.flatten()))[crop.size * 0.2]) & 
                     (crop <  (np.sort(crop.flatten()))[crop.size * 0.95]) )

    myrange = [np.median(crop)-10.*np.std(crop[mask]) , 
               np.median(crop)+10.*np.std(crop[mask])]
    junk = plt.hist(crop.flatten(), bins=50, range=myrange)
    mode = np.max((junk[1])[np.where(junk[0] == np.max(junk[0]))]) #approximately
    plt.clf()

    #calculate mask with bad pixels in place
    mask = np.where( (crop > (mode - 3.*np.std(crop[mask], ddof=1))) &
                     (crop < (mode + 3.*np.std(crop[mask], ddof=1))) )

    print (1.-float(crop[mask].size)/float(crop.size))*100., \
        "% of pixels were rejected as bad."
    
    #save mask for later
    masks += mask
    #pdb.set_trace()
    #check masking?
    if 1: 
        junk = plt.hist(crop.flatten(), bins=50, range=myrange)
        plt.title("Pixel Brightness Histogram, CDS "+str(max(files))+'-09')
        plt.xlabel('ADU')
        plt.ylabel('N_Occurences')
        plt.vlines( (np.min(crop[mask]) , np.max(crop[mask])) , 
                    0.1, 1.2*np.max(junk[0]), colors='r')
        plt.vlines( (mode) , 0.1, 1.2*np.max(junk[0]), colors='g')
        plt.show()
        
for i in range(len(files)): #cds frame delta
    #Update user on progress
    print str(int(round(float(i) / len(files) * 100.))) + '% done.'

    frame = pyfits.getdata(dir+filename_sub+str(files[i])+'.fits')
    cds = fluxdir*(frame-f9)

    #Define reference pixels
    ref = [0,1,2,3, len(cds)-1, len(cds)-2, len(cds)-3, len(cds)-4]
    #subtract median of reference pixels
    for j in range((cds.shape)[1]):
        cds[ : , j] -= np.median(cds[ref, j])

    for j in range(n_crops):
        crop  = cds[center+yshifts[j] : center+yshifts[j]+70 , 
                    center+1+(j-n_crops/2)*64 : center-1+(j-n_crops/2)*64 +64 ]
    
        #crosstalk correction increases the variance but does not affect the signal
        mask = masks[j*2:j*2+2]
        noisearr[i , j] = np.var(crop[mask], ddof=1) * correction
        fluxarr[i , j] = np.mean(crop[mask])

        #verify masking is being reasonable
        if i == len(files)-1:
            myrange = [np.median(crop)-10.*np.std(crop[mask]) , 
                       np.median(crop)+10.*np.std(crop[mask])]
            junk = plt.hist(crop.flatten(), bins=50, range=myrange)
            
            plt.title("Pixel Brightness Histogram, CDS "+str(max(files))+'-09')
            plt.xlabel('ADU')
            plt.ylabel('N_Occurences')
            plt.vlines( (np.min(crop[mask]) , np.max(crop[mask])) , 
                        0.1, 1.2*np.max(junk[0]), colors='r')
            plt.show()

for i in range(n_crops): #loop over n_crops
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

pdb.set_trace() #Returns to command line for debugging/additional commands


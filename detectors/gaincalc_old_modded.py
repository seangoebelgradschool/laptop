#/usr/bin/env python
#Updated 8/26/14 11:30 PM

# Plots the variance and flux levels vs CDS frame deltas in order to determine the read
# noise and gain.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

#H4RG
#dir='/home/H4RG/Data/20140102124546/'
#filename_sub = 'H4RG_R01_M01_N'
#f9 = pyfits.getdata(dir+filename_sub+'09.fits')
#crosstalk = 0.011

#H2RG
crosstalk = 0.005 #close enough to the 0.3 and 0.7% values
dir = 'ishell_science/'
filename_sub = 'ishell_flats_1-00040.Z.'
f9 = pyfits.getdata(dir+filename_sub+'8.fits') #For H2RG

n_crops =3 #Number of crops to check, should be integer

files = np.arange(9, 23, 1) #file list to loop over
center = ((f9.shape)[1])/2
yshifts = np.zeros(n_crops)

print "Crosstalk correction factor is", (str(crosstalk*100))[0:4], '%.'
correction = (5.*crosstalk + 1.) / (crosstalk + 1.)

#display crops for user to decide if they're good
#yshifts = [ 70, 140, 70]
masks = []
frame = pyfits.getdata(dir+filename_sub+str(np.max(files))+'.fits')
cds = 1.*(f9-frame)
for i in range(n_crops):
    answer = 'n'
    yshift = 0
    while (answer != 'y') :
        crop  = cds[center+yshift : center+70+yshift , 
                    center+1+(i-n_crops/2)*64 : center-1+(i-n_crops/2)*64 +64]
        
        plt.imshow(crop, interpolation='nearest', 
                   vmin=np.median(crop)-5.*np.sqrt(np.median(crop)),
                   vmax=np.median(crop)+5.*np.sqrt(np.median(crop)) )
        plt.title('Finding Crop '+str(i+1)+' of '+str(n_crops))
        plt.colorbar()
        plt.show()
        
        #ask user if the crop is acceptable
        answer = raw_input('Is the crop clean (y/n)? ')
        print
        if (answer.lower() != 'y'): yshift += 70
        if (yshift > center-71): yshift = -1.*center
        yshifts[i] = yshift

#for i in range(n_crops):
#    answer = 'n'
#    yshift = 0
#    yshift = yshifts[i]
#    crop  = (-1.*(f9-frame))[center+yshift : center+yshift+70 , 
#                             center+1+(i-n_crops/2)*64 +64*5: center-1+(i-n_crops/2)*64 +64 +64*5]
#    while (answer != 'y'):
#        crop  = (-1.*(f9-frame))[center+yshift : center+yshift+70 , 
#                             center+1+(i-n_crops/2)*64 +64*5: center-1+(i-n_crops/2)*64 +64 +64*5]
#        plt.imshow(crop, interpolation='nearest', 
#                   vmin=np.median(crop)-5.*np.sqrt(np.median(crop)),
#                   vmax=np.median(crop)+5.*np.sqrt(np.median(crop)) ) #show the frame
#        plt.title('Finding Crop '+str(i+1)+' of '+str(n_crops))
#        plt.colorbar()
#        plt.show()

        #ask user if the crop is acceptable
#        answer = raw_input('Is the crop clean (y/n)? ')
#        print
#        if (answer.lower() != 'y'): yshift += 70 #shift elsewhere in the column
#        if (yshift > center-71): yshift = -1.*center #don't go off the top of the image
#        yshifts[i] = yshift

    #crop = cds[center + int(yshifts[j])    : center + int(yshifts[j]) + 70 , 
    #           center+1 + (j-n_crops/2)*64 : center-1 + (j-n_crops/2)*64 + 64]
    #crop = cds[int(yshifts[j])    : int(yshifts[j]) + 70 , 
    #           center+1 + (j-n_crops/2)*64 : center-1 + (j-n_crops/2)*64 + 64]

    #maskcrop = mask_img[int(yshifts[j])    : int(yshifts[j]) + 70 , 
    #           center+1 + (j-n_crops/2)*64 : center-1 + (j-n_crops/2)*64 + 64]

    #Select central 60% of pixels, which hopefully don't include outlier populations
    mask = np.where( (crop >  (np.sort(crop.flatten()))[crop.size * 0.2]) & 
                     (crop <  (np.sort(crop.flatten()))[crop.size * 0.8]) )

    myrange = [np.median(crop)-10.*np.std(crop[mask]) , 
               np.median(crop)+10.*np.std(crop[mask])]
    junk = plt.hist(crop.flatten(), bins=50, range=myrange)
    mode = np.max((junk[1])[np.where(junk[0] == np.max(junk[0]))]) #approximately
    plt.clf()

    #calculate mask without bad pixels sorta filtered
    mask = np.where( (crop > (mode - 5.*np.std(crop[mask], ddof=1))) &
                     (crop < (mode + 5.*np.std(crop[mask], ddof=1))) )

    #mask = np.where(maskcrop == 1)

    print (1.-float(crop[mask].size)/float(crop.size))*100., \
        "% of pixels were rejected as bad."
    
    #save mask for later
    #masks[mask , i] = 1 #good pixels are indicated with a 1
    #pdb.set_trace()
    masks += mask
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
       

noisearr = np.zeros((len(files) , n_crops)) #stores variances
fluxarr = np.zeros((len(files) , n_crops)) #same dimensions as niosearr


for i in range(len(files)): #cds frame delta
    frame = pyfits.getdata(dir+filename_sub+str(files[i])+'.fits')
    cds  = 1.*(f9-frame)
    #cds /= flat
    #cds *= correction #correct for crosstalk FREAKING WRONG MATH NO NO NO
    if (i % 3 == 0): print str(int(round(float(i)/len(files)*100.))), '% done.'

    #Define reference pixels
    ref = [0,1,2,3, len(cds)-1, len(cds)-2, len(cds)-3, len(cds)-4]
    #subtract median of reference pixels
    for j in range((cds.shape)[1]):
        cds[ : , j] -= np.median(cds[ref, j])
    
    for j in range(n_crops): #crops within a frame
        crop = cds[center + int(yshifts[j])    : center + int(yshifts[j]) + 70 , 
                   center+1 + (j-n_crops/2)*64 : center-1 + (j-n_crops/2)*64 + 64]

        #if i == len(files)-1:
        #    plt.imshow(crop, interpolation='nearest', vmin=np.median(crop)-100,
        #               vmax=np.median(crop)+100)
        #    plt.colorbar()
        #    plt.title('Getting Analyzed!')
        #    plt.show()
            
#        #calculate mask with bad pixels in place
#        mask = np.where( (crop > np.median(crop) - 3.*np.std(crop, ddof=1)) &
#                         (crop < np.median(crop) + 3.*np.std(crop, ddof=1)) )
#        #rinse and repeat without bad pixels
#        mask = np.where( (crop > np.median(crop[mask]) - 3.3*np.std(crop[mask], ddof=1)) &
#                         (crop < np.median(crop[mask]) + 3.3*np.std(crop[mask], ddof=1)) )

        mask = masks[j*2:j*2+2]
        #pdb.set_trace()
        #check masking periodically
        if (files[i] % 20 == 0):
            myrange = [np.median(crop)-500 , np.median(crop)+500]
            junk = plt.hist(crop.flatten(), bins=500, range=myrange)
            plt.title("Pixel Brightness Histogram, CDS "+str(files[i])+'-09')
            plt.xlabel('ADU')
            plt.ylabel('N_Occurences')
            plt.vlines( (np.min(crop[mask]) , np.max(crop[mask])) , 0.1, 1e4, colors='r')
            plt.show()
            
        #crosstalk correction increases the variance but does not affect the signal
        noisearr[i, j] = np.var(crop[mask], ddof=1) * correction
        fluxarr[i, j] = np.mean(crop[mask])


#readnoise = np.zeros(n_crops) #stores calculated read noises
fits = np.zeros((3, n_crops)) #quadratic fitting terms for each of the crops
#flux_fits = np.zeros((2 , n_crops)) #linear fitting terms for each of the crops

for i in range(n_crops): #loop over n_crops
    plt.plot(fluxarr[:,i], noisearr[:,i], 'o')
    #plt.title('Crop '+ str(i+1) + ' of ' + str(n_crops) + '  Pixel Variance vs. CDS Frame Delta')
    plt.title('Pixel Flux vs. Variance')
    plt.xlabel('Flux [ADU]')
    plt.ylabel('Variance [ADU^2]')

    #fit a polynomial to the variance data
    coeffs = np.polyfit(fluxarr[:,i], noisearr[:,i], 2) #quadratic fit
    fits[ : , i] = coeffs #store coeffs
    fit = np.poly1d(coeffs)
    noise_ys = fit(fluxarr[:,i])
    plt.plot(fluxarr[:,i], noise_ys)

    #plt.plot(files-9, noise_ys)
    plt.text(fluxarr[3,i], 0.8*np.max(noisearr[:,i]), 
             'y='+str(coeffs[0])[0:6]+'x^2+'+ str(coeffs[1])[0:4]+'x+'+str(coeffs[2])[0:4])
    plt.show()
    

readnoise = np.sqrt(fits[2, : ]) #convert to ADU

print "Read noise values [ADU]:", readnoise
print "Mean [ADU]:", np.mean(readnoise)
print "Median [ADU]:",  np.median(readnoise)
print "Uncertainty (sigma/root(N)) [ADU]:", np.std(readnoise, ddof=1) / np.sqrt(n_crops)
print

#pdb.set_trace()


#NOW CALCULATE THE GAIN
gain = 1./fits[1, :] #mean signal / (shot noise)^2
print "Gain [e-/ADU] values:", gain
print "Mean gain [e-/ADU]:" , np.mean(gain)
print "Median gain [e-/ADU]:", np.median(gain)
print "(Incorrectly calculated) gain uncertainty [e-/ADU]:", np.std(gain, ddof=1) / np.sqrt(n_crops)

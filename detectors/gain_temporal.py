#/usr/bin/env python
#updated 6/2/16
#OLD LOCAL VERSION. LATEV TSERSION ON NGSTBIGRAM
#Calculates gain using temporal method (instead of spatial method)


import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

dir = 'h4rg/20160601/'
filename_sub = 'H4RG_R01_M01_N'

#populate cube
frames = np.arange(10, 40, 1)

for i in frames:
    print "reading frame", i, "of", np.max(frames)
    if i==np.min(frames):
        img = pyfits.getdata(dir+filename_sub+str(i)+'.fits')
        imcube = np.zeros((len(frames), np.shape(img)[0], np.shape(img)[1]))
        imcube[i-np.min(frames)] = img
    else:
        imcube[i-np.min(frames)] = pyfits.getdata(dir+filename_sub+str(i)+'.fits')

cdscube = imcube[1:, :, :] - imcube[:-1, :, :]

#Define reference pixels
ref = [0,1,2,3,
       np.shape(cdscube)[2]-1, np.shape(cdscube)[2]-2,
       np.shape(cdscube)[2]-3, np.shape(cdscube)[2]-4]
#subtract reference pixels at top and bottom
for z in range(np.shape(cdscube)[0]):
    print "Ref pixel correcting frame", z+np.min(frames), "of", np.max(frames)
    for x in range(np.shape(cdscube)[1]):
        cdscube[z, : , x] -= np.median(cdscube[z, ref, x])
        
#for i in frames:
    #update progress
#    if round((i-10)/30 % 10.) == 0: print str(int(round((i-10.)/30.*100)))+"% done."
    
    #cds1 = pyfits.getdata(dir+filename_sub+str(i+1)+'.fits') - \
    #       pyfits.getdata(dir+filename_sub+str(i)  +'.fits')


#    if 0: #(i == 10) or (i==38):
#        plt.hist(cds1.flatten(), bins=100, range=[100,700])
#        plt.show()
    
#    if (i==10): #first time
#        cube1 = cds1[4:1000,4:1000] #define cube
#    else:
#        cube1 = np.dstack((cube1, cds1[4:1000,4:1000])) #append cds frame onto cube

#284-314 for i=10
#270-300 for i=50

variance1 = np.var(cdscube, 0, ddof=1) #compute variance along z axis
#variance2 = np.var(cube2, 2, ddof=1) #compute variance along z axis
mean1 = np.median(cdscube, 0)

#mask = np.where((mean1 > 284) & (mean1 < 310) )
#mean1 = mean1[mask]
#variance1 = variance1[mask]

#mean2 = np.median(cube2, 2)

variance1 -= (7.15**2) #correct for read noise

gain1 = (mean1) / (variance1)
#gain2 = mean2/variance2

print "Mean 1:", np.median(mean1)
print "Variance 1:", np.median(variance1)
print "Gain 1:", np.median(gain1)
#print
#print "Mean 2:", np.median(mean2)
#print "Variance 2:", np.median(variance2)
#print "Gain 2:", np.median(gain2)

#plt.hist(gain.flatten(), bins=100 , range=[-1,5])
#plt.show()

#plt.plot(medians, 'o')
#pdb.set_trace()
selection = np.round(np.random.uniform(size=500)*np.size(mean1)).astype(int)
plt.plot((variance1.flatten())[selection], \
         (mean1.flatten())[selection], '.')
#plt.plot( variance2.flatten()[selection], mean2.flatten()[selection], 'r.')
plt.plot(np.arange(1e6), 2.9*np.arange(1e6))
plt.xlabel('variance')
plt.ylabel('mean')
#plt.axis((0, 3e5, -200, 600))
plt.xlim((0,1000))
plt.ylim((-20, 1000))
plt.show()

for i in range(100):
    y = np.random.uniform()*np.shape(cube1)[0]
    x = np.random.uniform()*np.shape(cube1)[0]
    plt.plot(cube1[y,x,:], 'o')
    plt.title(np.median(cube1[y,x,:]) / np.var(cube1[y,x,:], ddof=1) )
    plt.show()
    

pdb.set_trace()

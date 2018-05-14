#/usr/bin/env python
#updated 2/18/15

#Calculates gain using temporal method (instead of spatial method)


import pyfits
import matplotlib.pyplot as plt
import numpy as np
import pdb

dir = 'saphira/'
filename = '150324_163748.fits'
img = pyfits.getdata(dir+filename)

for i in np.arange(10, 80, 2):
    im0 = img[i, : , : ]
    im1 = img[i+19, : , : ]
    cds = im0-im1
    
    if (i==10): #first time
        cube = cds #define cube
    else:
        cube = np.dstack((cube, cds)) #append cds frame onto cube

#correct for nonlinearity
difref = 0
for i in range(cube.shape[2]):
    diff = np.median(cube) - np.median(cube[ :, :, i])
    cube[ :, :, i] += diff
    if (abs(diff) > difref): difref = abs(diff)
print "Max nonlinearity correction:", str(difref/np.median(cube)*100.)[0:4], "%"

variance = np.var(cube, 2, ddof=1) #compute variance along z axis
mean = np.median(cube, 2)

#mask = np.where((mean1 > 400) & (mean1 < 800) )
#mean1 = mean1[mask]
#variance1 = variance1[mask]

readnoise=9 #in ADU
gainguess=2.9 #e-/adu
variance -= ((readnoise/gainguess)**2) #correct for read noise

gain = mean / variance
#gain2 = mean2/variance2

plt.imshow(variance, vmin=0, vmax=2000, interpolation='none')
plt.colorbar()
plt.title('Variance'+filename)
plt.show()

#save fits image
hdu = pyfits.PrimaryHDU(variance)
hdu.writeto('saphira/'+filename[0:13]+'_var'+'.fits', clobber=True)
print "variance image saved."

print "Median:", np.median(mean)
print "Variance:", np.median(variance)
print "Gain:", np.median(gain)

plt.hist(gain.flatten(), bins=100 , range=[-1,5])
plt.show()

if (variance.size < 1000):
    selection = range(variance.size)
else:
    selection = (np.round(np.arange(0., variance.size, float(variance.size)/1000.))).astype(int)

answer = [0,0]
i=10
cols = [0+i, 32+i, 64+i, 96+i, 128+i, 160+i, 192+i, 224+i, 256+i, 288+i]
while len(answer) == 2:
    #plot a single channel
    plt.plot( (variance[:, cols]).flatten(), (mean[:, cols]).flatten(), '.')
    #plot all channels
    #plt.plot( (variance.flatten())[selection], (mean.flatten())[selection], '.')
    plt.plot(np.arange(1e6), 2.9*np.arange(1e6))
    plt.xlabel('variance')
    plt.ylabel('mean')
    plt.xlim((0,1000))
    plt.ylim((0, 500))
    plt.show()
    
    answer = tuple(int(x.strip()) for x in raw_input('Input coords of data point\
 to examine in format of x,y. 0 = skip. ').split(','))
    if len(answer) == 2:
        loc = np.where((variance - answer[0])**2 + (mean - answer[1])**2 == 
                       np.min((variance - answer[0])**2 + (mean - answer[1])**2))

        print np.median(cube[loc[0], loc[1], :].flatten())
        print mean[loc[0], loc[1]]
        print np.var(cube[loc[0], loc[1], :].flatten(), ddof=1) - \
            ((readnoise/gainguess)**2) #correct for read noise
        print variance[loc[0], loc[1]]

        plt.plot(cube[loc[0], loc[1], :].flatten(), 'o')
        plt.xlabel('CDS Frame #')
        plt.ylabel('ADU')
        plt.show()

#Create Flux vs Variance plots for each channel
varianceperchannel = np.array([])
for i in range(32):
    cols = [0+i, 32+i, 64+i, 96+i, 128+i, 160+i, 192+i, 224+i, 256+i, 288+i]
    var = np.median( (variance[:, cols]).flatten() )
    median = np.median((mean[:, cols]).flatten())
    varianceperchannel = np.append(varianceperchannel, var)

    plt.plot( (variance[:, cols]).flatten(), (mean[:, cols]).flatten(), '.')
    plt.plot(var, median, 'r*', markersize=10)
    plt.xlim((0,2000))
    plt.ylim((0, 500))
    plt.title('Channel '+ str(i))
    plt.xlabel('Variance [ADU^2]')
    plt.ylabel('Median Flux [ADU]')
    plt.text(25, 400, 'Median flux='+str(median)[0:4])
    plt.text(25, 375, 'Median variance='+str(var)[0:4])
    plt.text(25, 350, 'Gain=' + str(median/var)[0:4])
    plt.savefig('saphira/'+filename[0:13]+'_FluxVsVarianceChannel'+str(i)+'.png')
    #plt.show()
    plt.clf()
#Create Variance vs. Channel Number plot
plt.plot(varianceperchannel/np.median(varianceperchannel), 'o')
plt.title(filename)
plt.xlabel('Channel #')
plt.ylabel('Normalized Variance [ADU^2]')
plt.xlim((-1, 32))
plt.savefig('saphira/'+filename[0:13]+'_ChannelVsVariance.png')
plt.show()
#plt.clf()

pdb.set_trace()

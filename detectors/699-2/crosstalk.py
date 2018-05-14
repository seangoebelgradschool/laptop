#/usr/bin/env python
#Updated 4/3/14 9:15 PM

import pyfits
#import pylab as py
import matplotlib.pyplot as plt
import numpy as np

#fitsfile = '/home/H4RG/Data/20131211182735/H4RG_R04_M01_N420.fits'
dir = '/media/sean/SDXC/IRTF_Darks/dec10/'
#dir = '/media/sean/SDXC/IRTF_Darks/dec09/'

#Decomment for Dec 9 Data
#r1 = '12' #Dec 9
#r2 = '13' #Dec 9
#filename_sub = 'SpeX_SGA_darks-000' #Dec 9

#Decomment for Dec 10 Data
r1 = '30' #Dec 10
r2 = '31' #Dec 10
filename_sub = 'SpeX_SGA_DEC10-000' #Dec 10

R1_pt1 = pyfits.getdata(dir+filename_sub+r1+'.Z.72.fits')
R1_pt2 = pyfits.getdata(dir+filename_sub+r1+'.Z.200.fits')
cds_r01 = R1_pt2 - R1_pt1

R2_pt1 = pyfits.getdata(dir+filename_sub+r2+'.Z.72.fits')
R2_pt2 = pyfits.getdata(dir+filename_sub+r2+'.Z.200.fits')
cds_r02 = R2_pt2 - R2_pt1

cosmic_ray_frame = (cds_r02 - cds_r01)#[0:1000, 0:1000]

flattened = cosmic_ray_frame.flatten()
print len(flattened)
junk = plt.hist(flattened, bins=1000, log=True)


plt.ylim([0.1,1e6]) #formerly py
plt.title("Pixel Brightness Histogram")
plt.xlabel('ADU')
plt.ylabel('N_Occurences')
plt.show()

cr_loc = np.where(abs(cosmic_ray_frame) > 5e3)# | (cosmic_ray_frame > 5e3)) #locations of CRs
#loc_r05 = np.where(cosmic_ray_frame > 40e3) #locations of CRs in R05
#loc_r04 = np.where(cosmic_ray_frame < -40e3) #locations of CRs in R0'+r1+'

#for i in range(len(loc_r04[0])):
#    im=cds_r04[(loc_r04[0])[i]-5:(loc_r04[0])[i]+5, (loc_r04[1])[i]-5:(loc_r04[1])[i]+5]
#    plt.imshow(im, interpolation='none')
#x    plt.show()
#    print (loc[0])[i], (loc[1])[i]
#    print cosmic_ray_frame[(loc[0])[i], (loc[1])[i]]
#    print

#for i in range(len(loc_r05[0])):
#    im=cds_r05[(loc_r05[0])[i]-5:(loc_r05[0])[i]+5, (loc_r05[1])[i]-5:(loc_r05[1])[i]+5]
#    plt.imshow(im, interpolation='none')
#    plt.show()

#create average frame
try:
    avg_cr_frame = pyfits.getdata('avg_cr_frame_R0'+r2+'-R0'+r1+'.fits')
    print "Restored average cosmic ray frame."
except IOError:
    print "Building average cosmic ray frame..."
    avg_cr_frame = np.zeros((2048,2048))
    progress = 0. #for progress updater

    total_frames = (265. - 201. + 73. - 9.) * 2. #256

    for i in range(201, 265):
        avg_cr_frame -= pyfits.getdata(dir+filename_sub+r1+'.Z.'+str(i)+'.fits') #coadd next frame
        progress += 1.
        if (progress % round(total_frames/10.) == 0): 
            print str(int(round(progress/total_frames*100.)))+'% done.'
    for i in range(9, 73):
        #name = str(i)
        #while(len(name) < 2): name = '0'+name #make it the right num of characters
        avg_cr_frame += pyfits.getdata(dir+filename_sub+r1+'.Z.'+str(i)+'.fits')
        progress += 1.
        if (progress % round(total_frames/10.) == 0): 
            print str(int(round(progress/total_frames*100.)))+'% done.'
    for i in range(201, 265):
        avg_cr_frame += pyfits.getdata(dir+filename_sub+r2+'.Z.'+str(i)+'.fits')
        progress += 1.
        if (progress % round(total_frames/10.) == 0):
            print str(int(round(progress/total_frames*100.)))+'% done.'
    for i in range(9, 73):
        #name = str(i)
        #while(len(name) < 2): name = '0'+name #make it the right num of characters
        avg_cr_frame -= pyfits.getdata(dir+filename_sub+r2+'.Z.'+str(i)+'.fits')
        progress += 1.
        if (progress % round(total_frames/10.) == 0): 
            print str(int(round(progress/total_frames*100.)))+'% done.'

    avg_cr_frame /= total_frames
    pyfits.writeto('avg_cr_frame_R0'+r2+'-R0'+r1+'.fits', avg_cr_frame)
    print "Saved average cosmic ray frame."

#flattened = avg_cr_frame.flatten()
#junk = plt.hist(flattened, bins=1000, log=True)
#plt.ylim([0.1,1e7]) #formerly py
#plt.xlim([-60e3, 60e3])
#plt.title("Pixel Brightness Histogram (Avg Frame)")
#plt.xlabel('ADU')
#plt.ylabel('N_Occurences')
#plt.show()

summary = open('summary_R0'+r2+'-R0'+r1+'.txt', 'w')
summary.write('Ramp: ')
summary.write('File name:                    ')
summary.write('Mean Crosstalk (%): ')
summary.write('Center (ADU): ')
summary.write('Left (ADU): ')
summary.write('Left (% of center): ')
summary.write('Right (ADU): ')
summary.write('Right (% of center): ')
summary.write('Upper (ADU): ')
summary.write('Upper (% of center): ')
summary.write('Lower (ADU): ')
summary.write('Lower (% of center): \n')

j=1 #cosmic ray number
for i in range(len(cr_loc[0])):
    #Check if the CR is centered in the frame
    center = avg_cr_frame[(cr_loc[0])[i], (cr_loc[1])[i]]
    top    = avg_cr_frame[(cr_loc[0])[i], (cr_loc[1])[i]+1]
    bottom = avg_cr_frame[(cr_loc[0])[i], (cr_loc[1])[i]-1]
    right  = avg_cr_frame[(cr_loc[0])[i]+1, (cr_loc[1])[i]]
    left   = avg_cr_frame[(cr_loc[0])[i]-1, (cr_loc[1])[i]]

    #if difference btw right and left pixels is <0.1% of central peak
    if ( (abs(right - left) < 0.01 * abs(center)) &
         #if difference btw top and bottom pixels is <0.1% of central peak
         (abs(top - bottom) < 0.01 * abs(center)) ):
    
        #sanity check
        print "Center:",  center #abs( avg_cr_frame[(cr_loc[0])[i], (cr_loc[1])[i]])
        print "Right:", right #avg_cr_frame[(cr_loc[0])[i]+1, (cr_loc[1])[i]]
        print "Left:", left #avg_cr_frame[(cr_loc[0])[i]-1, (cr_loc[1])[i]]
        print "Top:", top #avg_cr_frame[(cr_loc[0])[i], (cr_loc[1])[i]+1]
        print "Bottom:", bottom #avg_cr_frame[(cr_loc[0])[i], (cr_loc[1])[i]-1]
        print

        #crop and show image
        im = avg_cr_frame[(cr_loc[0])[i]-4:(cr_loc[0])[i]+5, (cr_loc[1])[i]-4:(cr_loc[1])[i]+5]
        #if j==1:
        #    name = str(10)
        #    test = pyfits.getdata(dir+'H4RG_R0'+r2+'_M01_N'+name+'.fits')
        #    test = test[(cr_loc[0])[i]-4:(cr_loc[0])[i]+5, (cr_loc[1])[i]-4:(cr_loc[1])[i]+5]
        #    print test[2:7, 2:7]
        plt.imshow(im, interpolation='none')#nearest')
        plt.colorbar()
        plt.show()

        #peak = np.max(im)
        num=str(j)
        if len(num)==1: num='0'+num
        filename='centered_cr_'+num+'_R0'+r2+'-R0'+r1+'.fits'
        j+=1

        im.__idiv__(center / 100.) #scale pixels to percent of central peak
        pyfits.writeto(filename, im, clobber='True') #save fits file

        if (center > 0): summary.write('R0'+r2+'  ') #Ramps
        else: summary.write('R0'+r1 +'  ') #Ramps
		
        summary.write(filename + ' ') #file name
        summary.write(str(np.mean((right/center, left/center, top/center, bottom/center))*100.)
                      [0:5] + '               ') #mean crosstalk
        summary.write(str(center)[0:6]+'        ') #center ADU

        summary.write(str(left)[0:4]+'        ') #left ADU
        summary.write(str(left/center*100.)[0:4]+'                ') #left %

        summary.write(str(right)[0:4]+'         ') #right ADU
        summary.write(str(right/center*100.)[0:4]+'                 ') #right %

        summary.write(str(top)[0:4]+'         ') #top ADU
        summary.write(str(top/center*100.)[0:4]+'                 ') #top %

        summary.write(str(bottom)[0:4]+'         ') #bottom ADU
        summary.write(str(bottom/center*100)[0:4]+'\n') #bottom %

        print "Saved 9x9 array."
        #save file name, peak, avg cross talk to txt file?
    #else:
        #print "Nope, not close!"

summary.close() #stop writing to file
#home/.matplotlib/matplotlib.rc interpolationd default value should be changed

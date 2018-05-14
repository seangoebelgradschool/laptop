import numpy as np
import pyfits
import matplotlib.pyplot as plt
#from lmfit.models import LorentzianModel
import scipy.ndimage.interpolation
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import pdb

def main():
    image = pyfits.getdata('figs/biggerr/biggerr,2,6,1_collapsed.fits')
    loc = np.isnan(image)
    image[loc] = 0
    image = scipy.ndimage.interpolation.rotate(image, 22.4, reshape=False)
    image[loc] = np.nan

    #make r^2 mape for CHARIS Images
    scaling = np.zeros(np.shape(image))
    
    center = (np.shape(scaling) - np.array([1,1]))/2
    for x in range(np.shape(scaling)[1]):
        for y in range(np.shape(scaling)[0]):
            scaling[y, x] = (x-float(center[0]))**2+(y-float(center[1]))**2
    scaling /= np.median(scaling[100,:])
    #image *= scaling

    ymin = 90
    height = 15
    xmin = 37
    crop = image[ymin: ymin+height, xmin : -1*xmin]
    #image = image[10:25,:]
    
    #plt.imshow(image, interpolation='none', vmin=-0.5, vmax=1)
    #plt.colorbar()
    #plt.show()

    spinepos = np.array([])
    spine = np.array([])
    
    for shift in range(np.shape(crop)[1]):

        y = crop[:, shift]
        x = range(len(y)) #numer of points

        if max(np.isnan(y))==True: continue #is there data?
        
        #mod = LorentzianModel()
        #pdb.set_trace()
        #pars = mod.guess(y)#, x=x)
        #out = mod.fit(y, params=pars, x=x)
        #print(out.fit_report(min_correl=0.25))

        try:
            n = len(x)
            mean = x[np.squeeze(np.where(y==np.max(y)))] #sum(x*y)/n
            sigma = sum(y*(np.array(x)-mean)**2)/n
            
            popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma, np.min(y), 0])
            #break
        except RuntimeError:
            try:
                n = len(x)
                mean = x[np.squeeze(np.where(y==np.max(y)))] #sum(x*y)/n
                sigma = sum(y*(np.array(x)-mean)**2)/n
                
                loc = np.squeeze(np.where(y==np.max(y)))
                if loc-3 >= 0:
                    minindex = loc-3
                else:
                    minindex = 0
                if loc+3 < len(y):
                    maxindex = loc+3
                else:
                    maxindex = len(y)-1
                    
                popt,pcov = curve_fit(gaus4,x[minindex:maxindex],y[minindex:maxindex],p0=[1,mean,sigma, np.min(y)])

            except RuntimeError:
                print "failed at ", shift
                #plt.plot(x,y,'o',label='data')
                #plt.show()
                continue

        if (popt[1] < 2) or (popt[1] > len(crop)-2) :
            try:
                n = len(x)
                mean = x[np.squeeze(np.where(y==np.max(y)))] #sum(x*y)/n
                sigma = sum(y*(np.array(x)-mean)**2)/n
                
                loc = np.squeeze(np.where(y==np.max(y)))
                if loc-3 >= 0:
                    minindex = loc-3
                else:
                    minindex = 0
                if loc+3 < len(y):
                    maxindex = loc+3
                else:
                    maxindex = len(y)-1
                    
                popt,pcov = curve_fit(gaus4,x[minindex:maxindex],y[minindex:maxindex],p0=[1,mean,sigma, np.min(y)])

                if (popt[1] < 2) or (popt[1] > len(crop)-2) :
                    continue #give up, move on

            except RuntimeError:
                print "failed at ", shift
                #plt.plot(x,y,'o',label='data')
                #plt.show()
                continue
            
        spinepos = np.append(spinepos, shift)
        spine = np.append(spine, popt[1])

    spinepos += xmin #compensate for crop
    spine += ymin  #compensate for crop

    plt.imshow(image*scaling, interpolation='none', vmin=-0.5, vmax=1)
    #plt.colorbar()
    plt.plot(spinepos, spine, 'o', color='white')
    plt.xlim(xmin, np.shape(image)[1]-xmin)
    plt.ylim(ymin, ymin+height)
    plt.show()

    np.savetxt('spinefit.txt', np.vstack((np.transpose(spinepos),
                                          np.transpose(spine))).transpose())
    #pdb.set_trace()
    
def gaus(x,a,x0,sigma, x_0, x_1):
    return np.array(a*exp(-(x-x0)**2/(2*sigma**2))) + x_0 + x_1*np.array(x)

def gaus4(x,a,x0,sigma, x_0):
    return np.array(a*exp(-(x-x0)**2/(2*sigma**2))) + x_0

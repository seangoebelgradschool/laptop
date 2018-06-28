#Makes a prettier version of Ian's capacitance plot so we don't have an Excel
# version.
import matplotlib.pyplot as plt
#import matplotlib.axes as ax
import numpy as np
from scipy.optimize import curve_fit
import pdb
from scipy import exp

#load in data
v_ic,ic=np.transpose(np.loadtxt('iannodecapacitance.txt'))
ic *= 1e15 #convert from F to fF

#load avalanche gain data. 0.5V is dodgy
v_ag,ag=np.transpose(np.loadtxt('avgaindata.txt'))
    
#load avalanche gain measurements
v_cg, n_adu_cg, var_cg = np.loadtxt('cgaindata.txt')

#these define where the transition between the functions occurs
thresh1 = 3 #defines V for start of exponential fit
thresh2 = 5 #defines V for end of quadratic fit

def main(excludefirstcg=True, ian=True, save=False,steps=10,scans=1):
##### PART 1: Get estimate of node capacitance
    #get initial guess of node capacitances by fitting to ian's data
    popt_c,pcov = curve_fit(expdec, v_ic, ic, p0=[26, 2, 30])

    dispersion = propagate(popt_c, excludefirstcg)
    initial_dispersion = dispersion
    bestfit = popt_c

    grid1 = np.linspace(popt_c[0] * 0.5, popt_c[0] * 2, steps)
    grid2 = np.linspace(popt_c[1] * 0.5, popt_c[1] * 2, steps)
    grid3 = [32]#np.linspace(popt_c[2] * 0.9, popt_c[2] * 1.1, steps/2)

    for aa in grid1:
        print str(int(round(float(np.where(grid1==aa)[0])/len(grid1)*100.)))+ '% done'
        for bb in grid2:
            for cc in grid3:
                newdispersion = propagate([aa,bb,cc], excludefirstcg)

                if newdispersion < dispersion: #have we found a better fit?
                    dispersion=newdispersion
                    bestfit = [aa,bb,cc]

    if (bestfit[0] == np.min(grid1)) or (bestfit[0] == np.max(grid1)):
        print "Best value is on edge of grid 1 space."
        print bestfit[0]
        print grid1
        print
    if (bestfit[1] == np.min(grid2)) or (bestfit[1] == np.max(grid2)):
        print "Best value is on edge of grid 2 space."
        print bestfit[1]
        print grid2
        print
    if (bestfit[2] == np.min(grid3)) or (bestfit[2] == np.max(grid3)):
        print "Best value is on edge of grid 3 space."
        print bestfit[2]
        print grid3
        print

    print "initial fit", popt_c
    print "bestfit:", bestfit
    print "initial dispersion, best dispersion", initial_dispersion, dispersion

            
def propagate(popt_c, excludefirstcg):
#### PART 2: Get capacitance-corrected avalanche gain

    #calculate avalanche gain corrected for capacitance
    ag_cor = ag * expdec(v_ag, popt_c[0], popt_c[1], popt_c[2])
    ag_cor /= ag_cor[1] #normalize to av. gain = 1 at 1.5V bias
    #add extra point at 0,1 to make the curve flatter
    ag_cor2 = np.append([1], ag_cor) #avalanche gain of 1 at 0v
    v_ag2 = np.append([0], v_ag) #avalanche gain of 1 at 0v

    popt_a,pcov = curve_fit(combofit, v_ag2, ag_cor2, p0=[0.07, -0.2, 1.1, 2.8, 2.1])

### PART 3: COMPUTE CHARGE GAIN
    cg = n_adu_cg / var_cg * combofit(v_cg, popt_a[0], popt_a[1], popt_a[2], \
                                      popt_a[3], popt_a[4])

    if excludefirstcg:
        #remove aberrant first point
        #v_cg2 = v_cg[1:]
        cg2 = cg[1:]
    else:
        cg2 = cg
    
    return np.std(cg2, ddof=1)



def orig(excludefirstcg=True, ian=True, save=False):
    #load Ian's measured node capacitance to use as initial guess for C(V)
    v_ic,ic=np.transpose(np.loadtxt('iannodecapacitance.txt'))
    ic *= 1e15 #convert from F to fF

    #load avalanche gain data. 0.5V is dodgy
    v_ag,ag=np.transpose(np.loadtxt('avgaindata.txt'))
    
    #load avalanche gain measurements
    v_cg, n_adu_cg, var_cg = np.loadtxt('cgaindata.txt')

    plt.figure(1, figsize=(7,10), dpi=100)
    if ian:
        plt.suptitle("Using Ian Baker's Measured Capacitances", fontsize=16)
    else:
        plt.suptitle("Using Capacitances that Minimize Charge Gain Dispersion", \
                     fontsize=16)
    
##### PART 1: Get estimate of node capacitance

    #get initial guess of node capacitances by fitting to ian's data
    popt,pcov = curve_fit(expdec, v_ic, ic, p0=[26, 2, 10])

#HARD CODE, BAD FORM,TIRED
    #popt = [52.79, 0.727, 15.297] #good, not fitting for first point
    #popt = [14.56, 1.259, 61.19] #good, fitting for first point
    popt = [13.2, 0.917, 32]
    
    #print "exponential decay: ", popt
    #fit 2nd order polynomial
    #fit = np.polyfit(v, c, 2)
    #print "2nd order polynomial:", fit
    #p = np.poly1d(fit)

    plt.subplot(3,1,1)
    plt.plot(v_ic, ic, 'o', label="Ian Baker's Values")#, markersize=12)

    x = np.linspace(0, 12, 1000)
    
    plt.plot(x,expdec(x, popt[0], popt[1], popt[2]), \
             label=r'$C (V) = ' + str(popt[0])[:6] + \
                             r'e^{-V/'+str(popt[1])[:6]+'} + '+ \
             str(popt[2])[:6]+'$')
    #plt.plot(x, p(x), label='2nd order poly')
    
    plt.title("Node Capacitance vs. Bias Voltage", fontsize=14)
    #plt.xlabel("Bias Voltage [V]", fontsize=12)
    plt.ylabel("Capacitance [fF]", fontsize=12)
    #plt.tick_params( labelsize=14)
    plt.xlim(0,12)
    plt.legend(fontsize=14)

#### PART 2: Get capacitance-corrected avalanche gain

    #calculate avalanche gain corrected for capacitance
    #plt.plot(v_ag, ag*40., 'ro', label='Cap. Uncorrected')
    ag *= expdec(v_ag, popt[0], popt[1], popt[2])
    ag /= ag[1] #normalize to av. gain = 1 at 1.5V bias
    #add in a point at [0,1]
    ag = np.append([1], ag) #avalanche gain of 1 at 0v
    v_ag = np.append([0], v_ag)
    
    #global thresh1, thresh2
    #these define where the transition between the functions occurs
    #thresh1 = 3 #defines V for start of exponential fit
    #thresh2 = 5 #defines V for end of polynomial fit
    #loc1 = np.where(v_ag < thresh2) #where polynomial is fit
    #loc1 = np.where(v_ag > thresh1) #where exponential is fit
    
    #fit 2nd order polynomial
    #fit1 = np.polyfit(v_ag[:5], ag[:5], 2)
    #print "2nd order polynomial:", fit1
    #fit exponential
    #p = np.poly1d(fit1)
    #popt1,pcov = curve_fit(expon2, v_ag[3:], ag[3:], p0=[2, 2])
    #print popt
    #combine the two fits
    popt,pcov = curve_fit(combofit, v_ag, ag, \
                          p0=[0.07, -0.2, 1.1, 2.8, 2.1])
    #print popt

    plt.subplot(3,1,2)
    plt.plot(v_ag, ag, 'ro')#, label='Capacitance-Corrected Avalanche Gain')
    #plt.plot(x, p(x))
    #plt.plot(x, expon2(x, popt1[0], popt1[1]))
    loc1 = np.where(x < thresh1)
    loc2 = np.where((x >= thresh1) & (x <= thresh2))
    loc3 = np.where(x > thresh2)
    plt.plot(x[loc1], combofit(x[loc1], popt[0], popt[1], popt[2], \
                               popt[3], popt[4]), 'b-', \
             label=r'$A = '+str(popt[0])[:5]+'V^{\;2}+'+ \
             str(popt[1])[:5] + 'V+' + str(popt[2])[:5]+'$'   )
    plt.plot(x[loc2], combofit(x[loc2], popt[0], popt[1], popt[2], \
                               popt[3], popt[4]), 'c-', \
             label="transition region")
    plt.plot(x[loc3], combofit(x[loc3], popt[0], popt[1], popt[2], \
                               popt[3], popt[4]), 'g-', \
             label=r'$A = 2^{(V-'+str(popt[3])[:5]+')/'+str(popt[4])[:5]+'}$')

    #plt.plot([thresh1, thresh1], [0, 100], 'm--')
    #plt.plot([thresh2, thresh2], [0, 100], 'm--')
    plt.legend(loc=2, fontsize=14)
    plt.yscale('log')
    plt.title("Capacitance Corrected Avalanche Gain", fontsize=14)
    #plt.xlabel("Bias Voltage")
    plt.ylabel("Avalanche Gain")
    plt.ylim(0.9,20)

### PART 3: COMPUTE CHARGE GAIN
    cg = n_adu_cg / var_cg * combofit(v_cg, popt[0], popt[1], popt[2], \
                         popt[3], popt[4])

    plt.subplot(3,1,3)
    plt.plot(v_cg, cg, 'o')
    if excludefirstcg:
        #remove aberrant first point
        v_cg = v_cg[1:]
        cg = cg[1:]
    plt.plot([np.min(x), np.max(x)], [np.mean(cg) , np.mean(cg)], \
             label='Mean = '+ str(np.mean(cg))[:5] + ' e-/ADU')
    plt.plot([np.min(x), np.max(x)], \
             [np.mean(cg) - np.std(cg, ddof=1), np.mean(cg) - np.std(cg, ddof=1)],\
             'r--',
             label=r'$\sigma=' + str(np.std(cg, ddof=1))[:5] + '$ e-/ADU')
    plt.plot([np.min(x), np.max(x)], \
             [np.mean(cg) + np.std(cg, ddof=1), np.mean(cg) + np.std(cg, ddof=1)],\
             'r--')
    
    plt.legend(fontsize=14, loc=0, framealpha=0.7)
    if excludefirstcg:
        plt.title("Charge Gain (0.5V point excluded)", fontsize=14)
    else:
        plt.title("Charge Gain", fontsize=14)
        
    plt.xlabel("Bias Voltage")
    plt.ylabel('Charge Gain [e-/ADU]')
    
    plt.subplots_adjust(top=0.92, bottom=0.06, left=0.10, right=0.97, hspace=0.26)

    if save:
        if ian:
            outtitle='Everything at Once, Ian Capacitance.png'
        else:
            outtitle='Everything at Once, Calculated Capacitance.png'
        
        plt.savefig(outtitle, dpi=150)
        print "Wrote", outtitle
        plt.clf()
    else:
        plt.show()
    

def expdec(x,a,tau, rho_0):
    return a*np.array(exp(-1. * np.array(x) / tau)) + rho_0

#def expon(x,a,tau):
#    return a*np.array(exp(np.array(x) / tau))

def expon2(x, d, e):
    return 2**((np.array(x)-d)/e)

def combofit(x, a, b, c, d, e):
    #fits ax^2 + bx + c below thresh1
    #fits 2^((x-d)/e) above thresh2
    #computes mean on sliding scale between thresh1 and thresh2
    x = np.array(x)
    y = np.zeros(len(x))
    if np.min(x) <= thresh1:
        loc = np.where(x<=thresh1)
        y[loc] = a*(x[loc])**2 + b*(x[loc]) + c
    if np.max(x) >= thresh2:
        loc = np.where(x>=thresh2)
        y[loc] = 2**((x[loc]-d)/e)
    if np.max((x>thresh1) & (x<thresh2)) == True:
        #for points between the thresholds, compute weighted average
        # of the two functions
        loc = np.where((x>thresh1) & (x<thresh2))
        scaling = (x[loc] - thresh1) / (thresh2-thresh1)
        y[loc] =  (a*(x[loc])**2 + b*(x[loc]) + c) * (1-scaling) + \
                  (2**((x[loc]-d)/e)) * scaling
    return y

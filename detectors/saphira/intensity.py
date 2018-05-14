import numpy as np
import pdb
import matplotlib.pyplot as plt

def main(a, b, theta, plot=False):

    #check that sizes are equivalent
    if np.size(a) != np.size(b) :
        print "a and b are different sizes!"
        return
        
    I = a**2 + b**2
    
    if np.size(a) == 1 :
        print I
        return #no point in continuing

    if plot != False:
        plt.subplot(311)
        plt.plot(a, label='a')
        plt.title('a')
        
        plt.subplot(312)
        plt.plot(b, label='b')
        plt.title('b')
        
        plt.subplot(313)
        plt.plot(I, label='I')
        plt.title('I')
        #plt.legend(loc=0)
        plt.show()

    theta_interval = theta[1:] - theta[:-1]
    
    dIdt = (I[1:] - I[:-1])/theta_interval
    d2Idt2 = (dIdt[1:] - dIdt[:-1])/theta_interval[:-1]

    dadt = (a[1:] - a[:-1])/theta_interval
    d2adt2 = (dadt[1:] - dadt[:-1])/theta_interval[:-1]
    
    dbdt = (b[1:] - b[:-1])/theta_interval
    d2bdt2 = (dbdt[1:] - dbdt[:-1])/theta_interval[:-1]

    #trim arrays to same length
    a = a[1:-1]
    dadt = dadt[1:]
    b = b[1:-1]
    dbdt = dbdt[1:]
    
    
    var_dIdt = np.var(dIdt, ddof=1)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    #var_dadt = np.var(dadt, ddof=1)
    #var_dbdt = np.var(dbdt, ddof=1)

    print "var_dIdt:", var_dIdt
    print "right side", var_a * np.mean(2 * dadt + \
                                        2 * a * d2adt2)**2 +\
                        var_b * np.mean(2 * dbdt + \
                                        2 * b * d2bdt2)**2 
    #print "var_dadt:", var_dadt
    #print "var_dbdt:", var_dbdt

    pdb.set_trace()

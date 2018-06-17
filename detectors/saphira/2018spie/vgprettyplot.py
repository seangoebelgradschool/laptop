import numpy as np
import matplotlib.pyplot as plt

#Loads volt gain data from a .npz file and then puts two plots
#next to each other to be eye candy in Sean's 2018 SPIE paper

def main(place='blah', save=False):
    if place=='blah':
        print "Please select place='lab' or place='scexao'"
        print "save=false or save=true are also options."
        return

    plt.figure(figsize=(10, 5), dpi=100)
    
    if place=='lab':
        suptitle = 'Laboratory Volt Gain Measurements'
    else:
        suptitle = 'SCExAO Volt Gain Measurements'
        
    plt.suptitle(suptitle, fontsize=18)#, y=1.005)
    for i in range(2):
        if place=='lab':
            if i==0:
                filename = 'Volt Gain with Mk15 SAPHIRA M09105-27 in JK Henriksen Mount.npz'
                title2 = 'JK Henriksen Mount'
            else:
                filename = 'Volt Gain with Mk15 SAPHIRA M09105-27 with Cryo Preamp.npz'
                title2 = 'ANU Preamplifier'
        else:
            if i==0:
                filename = 'SCExAO Volt Gain no Preamp.npz'
                title2 = 'JK Henriksen Mount'
            else:
                filename = 'SCExAO Volt Gain with Preamp.npz'
                title2 = 'ANU Preamplifier'
                
        savedstuff = np.load(filename)
        mytitle = savedstuff['arr_0']
        volts = savedstuff['arr_1']
        avg_adus = savedstuff['arr_2']
        coeffs = savedstuff['arr_3']
        p_volts = savedstuff['arr_4']

        plt.subplot(1,2,i+1)
        plt.plot(volts, avg_adus, 'o')
        plt.plot(volts, p_volts, '-', label=str(1.e6 / coeffs[0])[1:6] + \
                 ' ' + r'$\mu$'+'V/ADU')
        plt.title(title2)
        plt.xlabel('PRV [V]')
        plt.ylabel('Mean ADUs')
        plt.legend(loc=1)

        ax = plt.gca()
        ax.set_autoscale_on(False)
        plt.xlim(np.min(volts)-0.005 , np.max(volts)+0.005)
        myrange = np.max(avg_adus) - np.min(avg_adus)
        plt.ylim(np.min(avg_adus)-myrange*0.05 , np.max(avg_adus)+myrange*0.05)

    plt.subplots_adjust(left=0.09, right=0.99, wspace=0.27, top=0.87)

    outtitle='vg_'+place+'.png'
    if save:
        plt.savefig(outtitle, dpi=150)#, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()
    
    

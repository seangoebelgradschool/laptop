import numpy as np
#import pyfits
import matplotlib.pyplot as plt
import pdb

#plots histograms of what fraction of each value for each parameter
# produced "acceptably fitting" chi^2 fits.

def main(save=False):

    #set variables to plot against each other
    vars = ['R0', 'ALP_I', 'ALP_O', 'G', 'KSI0', 'BETA']
    dir = '20180627/'
    data = np.loadtxt(dir+'grid_search_stats.txt', dtype='str')
    chisq_accept = 0.8833 #threshold for acceptable fit
    
    for i in range(len(vars)):
        col = np.squeeze(np.where(data[0,:] == vars[i])) #eg 9
        values = np.unique(data[1:,col]).astype('float')
        if i==0:
            params = [values]
        else:
            params.append(values) #create list of all values for all params

    chi_col_index = np.squeeze(np.where(data[0,:] == 'CHISQ/dof'))

    plt.figure(figsize=(7.5, 10), dpi=100)
    
    for i in range(len(params)): #same as for var in vars
        xparam = vars[i] #eg 'r0'
        
        #number of values for that param, value and n_good and n_occurances
        tabulation = np.zeros((len(params[i]) , 3))
        tabulation[:, 0] = params[i] #first columns is the values

        col = np.squeeze(np.where(data[0,:] == vars[i])) #e.g. 9
        for j in range(1, np.shape(data)[0]): #go down column
            loc = np.where(tabulation[:,0] == float(data[j, col]))
            tabulation[loc, 2] += 1 #add to n_occurences
            if float(data[j, chi_col_index]) <= chisq_accept:
                tabulation[loc, 1] += 1 # add to n_good


        if xparam == 'R0': tit1 = r'$r_0$'
        elif xparam == 'ALP_I': tit1 = r'$\alpha_{in}$'
        elif xparam == 'ALP_O': tit1 = r'$\alpha_{out}$'
        elif xparam == 'BETA': tit1 = r'$\beta$'
        elif xparam == 'KSI0': tit1 = r'$\xi$'
        elif xparam == 'G': tit1 = r'$g$'

        plt.subplot(len(vars)/2, 2, i+1)
        plt.bar(range(len(tabulation[:,0])), tabulation[:,1] / tabulation[:,2],
                align='center', width=0.1*len(params[i]))#, color='black')
        if xparam=='R0':
            params[i] = np.round(params[i]*1.069).astype(int).astype(str)
            #for j in range(len(params[i])):
            #    params[i][j] = params[i][j][:2]
        plt.xticks(range(len(params[i])) , params[i], fontsize=14)
        plt.yticks(fontsize=14)
        #label=tabulation[:,0])

        #plt.title('Fraction of '+ tit1 + ' values that produced '+
        #          r'$\chi_{\nu}^{2} \leq$'+str(chisq_accept)[:6], fontsize=21)
        #plt.ylabel('Fraction', fontsize=21)
        plt.xlabel(tit1+' Value', fontsize=21)
        if i==2:
            plt.ylabel('Fraction', fontsize=21)

    plt.suptitle('Fraction of values that produced '+
                 r'$\chi_{\nu}^{2} \leq$'+str(chisq_accept)[:6], fontsize=21)
    plt.subplots_adjust(left=0.14, bottom=0.08, right=0.99, top=0.92, hspace=0.31, wspace=0.28)
    if save==True:
        plt.savefig('figs/histograms.png', dpi=150)
        plt.close()
    else:
        plt.show()
            

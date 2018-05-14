import numpy as np
#import pyfits
import matplotlib.pyplot as plt
import pdb

def main(save=True):
    #data for 20180404 reduction
    #vars = ['R0', 'ALP_I', 'ALP_O', 'KSI0', 'G', 'E']
    #r0 = [63, 73, 83] #73-83
    #alp_i = [2.5, 5, 7.5] #5-7.5
    #alp_o = [-2.5, -5] #-2.5
    #ksi0 = [1.5, 2, 3] #1.5
    #g = [0.2, 0.4, 0.6] #0.6
    #e = [0, 0.06, 0.1] #doesn't matter
    #params = [r0, alp_i, alp_o, ksi0, g, e]

    #set variables to plot against each other
    vars = ['R0', 'ALP_I', 'ALP_O', 'G', 'KSI0', 'BETA']
    dir = '20180503/'

    data = np.loadtxt(dir+'grid_search_stats_20180503.txt', dtype='str')

    for i in range(len(vars)):
        col = np.squeeze(np.where(data[0,:] == vars[i])) #eg 9
        values = np.unique(data[1:,col]).astype('float')
        if i==0:
            params = [values]
        else:
            params.append(values)

    chi_col_index = np.squeeze(np.where(data[0,:] == 'CHISQ/dof'))
    
    for i in range(np.shape(params)[0]-1):
        for j in range(i+1, np.shape(params)[0]):
            xparam = vars[i] #eg 'r0'
            yparam = vars[j] #eg 'alp_i'

            xcol = np.squeeze(np.where(data[0,:] == vars[i])) #eg 9
            ycol = np.squeeze(np.where(data[0,:] == vars[j])) #eg 3

            im = np.zeros(( len(params[j]) , len(params[i]) ))

            for x in range(len(params[i])):
                for y in range(len(params[j])):
                    loc = np.where( (data[1:,xcol].astype('float')==params[i][x]) &
                                    (data[1:,ycol].astype('float')==params[j][y]) ) 
                    loc = np.squeeze(np.array(loc))
                    
                    im[y,x] = np.mean(data[loc+1, chi_col_index].astype('float'))

            if xparam == 'R0': tit1 = r'$r_0$'
            elif xparam == 'ALP_I': tit1 = r'$\alpha_{in}$'
            elif xparam == 'ALP_O': tit1 = r'$\alpha_{out}$'
            elif xparam == 'BETA': tit1 = r'$\beta$'
            elif xparam == 'ksi0': tit1 = r'$\xi$'
            elif xparam == 'G': tit1 = r'$g$'
            
            if yparam == 'R0': tit2 = r'$r_0$'
            elif yparam == 'ALP_I': tit2 = r'$\alpha_{in}$'
            elif yparam == 'ALP_O': tit2 = r'$\alpha_{out}$'
            elif yparam == 'BETA': tit2 = r'$\beta$'
            elif yparam == 'ksi0': tit2 = r'$\xi$'
            elif yparam == 'G': tit2 = r'$g$'
                    
            plt.imshow(im, interpolation='none')
            plt.xticks(range(len(params[i])) , params[i])
            plt.yticks(range(len(params[j])) , params[j])
            plt.title(tit1+' vs. '+tit2+" mean " + r'$\chi_{\nu}^{2}$', fontsize=21)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1, top=0.93) #make space for title
            plt.xlabel(tit1, fontsize=21)
            plt.ylabel(tit2, fontsize=21)
            plt.colorbar()

            if save==True:
                plt.savefig(dir+'contour '+xparam+' vs '+yparam+'.png')
                plt.close()
            else:
                plt.show()

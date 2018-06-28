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

    #data = np.loadtxt('grid_search_stats.txt', dtype='str')

    data = np.loadtxt('grid_search_stats.txt', dtype='str')

    #data for 20180406 reduction
    vars = ['R0', 'ALP_I', 'ALP_O', 'KSI0', 'G']
    for i in range(len(vars)):
        col = np.squeeze(np.where(data[0,:] == vars[i])) #eg 9
        values = np.unique(data[1:,col]).astype('float')
        if i==0:
            params = [values]
        else:
            params = [params, values]
    
    #r0 = [63, 73, 83] #73-83
    #alp_i = [2.5, 5, 7.5] #5-7.5
    #alp_o = [-2.5, -5] #-2.5
    #ksi0 = [1.5, 2, 3] #1.5
    #g = [0.2, 0.4, 0.6] #0.6
    #e = [0, 0.06, 0.1] #doesn't matter

    #params = [r0, alp_i, alp_o, ksi0, g, e]


    
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

            plt.imshow(im, interpolation='none')
            plt.xticks(range(len(params[i])) , params[i])
            plt.yticks(range(len(params[j])) , params[j])
            plt.title(xparam+' vs. '+yparam+" mean chi^2")
            plt.xlabel(xparam)
            plt.ylabel(yparam)
            plt.colorbar()

            if save==True:
                plt.savefig('contour '+xparam+' vs '+yparam+'.png')
                plt.close()
            else:
                plt.show()

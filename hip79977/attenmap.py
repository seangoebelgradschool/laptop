import pyfits
import numpy as np
import pdb
from shutil import copyfile

#creates a map of how the forward modeling attenuates the disk model
#mostly unneeded now since charis_klip_grid_fwdmod produces the
#3D attenuation cube. Only use the last couple lines to apply attenuation
# to _cal fits file.

def main():
    fn_inputmodel  = 'figs/biggerr/egrater_0.600000,1.00000,4,-3.50000,112.400,0,0,112.400,84.6000,70,2,0_inputcolsc.fits'
    fn_psfsubmodel = 'figs/biggerr/egrater_0.600000,1.00000,4,-3.50000,112.400,0,0,112.400,84.6000,70,2,0_psfsubcol.fits'
    fn_realdata = 'figs/biggerr/biggerr,2,6,1_collapsed.fits'

    print "Input model:", fn_inputmodel
    print "PSF subtracted model:", fn_psfsubmodel
    
    inputmodel = pyfits.getdata(fn_inputmodel)
    #inputmodel /= np.nansum(inputmodel)
    
    psfsubmodel = pyfits.getdata(fn_psfsubmodel)
    #psfsubmodel /= np.nansum(psfsubmodel)

    #pdb.set_trace()
    
    atten = psfsubmodel / inputmodel
    atten[abs(atten) > 1] = 0
    outname = 'figs/biggerr/attenuationmap.fits'
    
    pyfits.writeto(outname, atten, clobber=True)
    print "Wrote:", outname

    fn_attencorrected = fn_realdata[:-5]+'_atten_corr.fits'
    copyfile(fn_realdata, fn_attencorrected) #duplicate file with new name

    realdisk = pyfits.getdata(fn_realdata)
    realdisk_corrected = realdisk / atten
    pyfits.update(fn_attencorrected, realdisk_corrected, ext=1)
    print "Wrote:", fn_attencorrected

#Makes a prettier version of Ian's capacitance plot so we don't have an Excel
# version.
import matplotlib.pyplot as plt
#import matplotlib.axes as ax
import numpy as np
import scipy
import pdb

v = [9,
     8.5,
     7.5,
     6.5,
     5.5,
     4.5,
     3.5,
     2.5,
     1.5,
     1]

c = [3.05E-14,
     3.07E-14,
     3.10E-14,
     3.14E-14,
     3.21E-14,
     3.32E-14,
     3.48E-14,
     3.73E-14,
     4.13E-14,
     4.65E-14]

c = np.array(c) * 1e15 #convert from F to fF

#coeffs = np.polyfit(v, c,2)
#fit = np.poly1d(coeffs)
#cfit = fit(v)

#scipy.optimize.curve_fit(lambda t, a, b:a*np.exp(b*t), v,c)

#pdb.set_trace()

plt.figure(1, figsize=(10,6), dpi=100)
plt.plot(v, c, 'o', markersize=12)
#plt.plot(v, cfit)
plt.title("Integrating Node Capacitance vs. Bias Voltage for Mk. 13 SAPHIRA", fontsize=16)
plt.xlabel("Bias Voltage [V]", fontsize=14)
plt.ylabel("Capacitance [fF]", fontsize=14)
plt.tick_params( labelsize=14)
plt.xlim(0,10)
plt.show()

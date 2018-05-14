import numpy as np
import matplotlib.pyplot as plt

#FIGURE OUT EQUIVALENT EXPOSURE TIME FOR CONSTANT S/N
#H2 info
qe_h2 = 0.55
grat_h2 = 1.
t_h2 = 900. #seconds
rn_h2 = 17.7
d_h2 = 0.06 #per second

#H2RG info
qe_h2rg = 0.83
grat_h2rg = 1#2.6 #in J band
rn_h2rg = 14.2
d_h2rg = 0.022 #per second
t_h2rg = np.arange(900)

signal_h2 = qe_h2 * grat_h2 * t_h2 *(2./9.)
signal_h2rg = qe_h2rg * grat_h2rg * t_h2rg * (2./9.)

shot_h2 = np.sqrt(signal_h2)
shot_h2rg = np.sqrt(signal_h2rg)

sn_h2 = signal_h2 / np.sqrt(rn_h2**2 + d_h2*t_h2 + shot_h2**2)
sn_h2rg = signal_h2rg / np.sqrt(rn_h2rg**2 + d_h2rg*t_h2rg + shot_h2rg**2)

print sn_h2

loc = np.where(abs(sn_h2rg - sn_h2) == min(abs(sn_h2rg - sn_h2)))
print "The H2RG can obtain an observation equivalent to an H2 900s J band integration in", t_h2rg[loc], 'seconds.'

plt.plot(t_h2rg, sn_h2rg)
plt.plot(t_h2rg, np.zeros(len(t_h2rg))+sn_h2)
plt.xlabel('Seconds')
plt.show()


#NOW FIGURE OUT THE S/N IMPROVEMENT FOR IDENTICAL EXPOSURE TIMES
t_h2rg = 900 #now both 900s
signal_h2 = qe_h2 * grat_h2 * 200.
signal_h2rg = qe_h2rg * grat_h2rg * 200.

shot_h2 = np.sqrt(signal_h2)
shot_h2rg = np.sqrt(signal_h2rg)

sn_h2 = signal_h2 / np.sqrt(rn_h2**2 + d_h2*t_h2 + shot_h2**2)
sn_h2rg = signal_h2rg / np.sqrt(rn_h2rg**2 + d_h2rg*t_h2rg + shot_h2rg**2)

print "H2 S/N", sn_h2
print "H2RG S/N", sn_h2rg

print "The H2RG is", sn_h2rg / sn_h2, "times better than the H2."


#NOW MAKE A PLOT SHOWING THE S/N IMPROVEMENT FOR J,H,K BANDS FOR VARIOUS H2 S/Ns
t=900 #s
flux = np.arange(300)/100. #photons/s

#J band first
qe_h2 = 0.55
qe_h2rg = 0.83
signal_h2 = qe_h2 * flux * t
signal_h2rg = qe_h2rg * flux * t

shot_h2 = np.sqrt(signal_h2)
shot_h2rg = np.sqrt(signal_h2rg)

sn_h2 = signal_h2 / np.sqrt(rn_h2**2 + d_h2*t + shot_h2**2)
sn_h2rg = signal_h2rg / np.sqrt(rn_h2rg**2 + d_h2rg*t + shot_h2rg**2)

plt.plot(sn_h2, sn_h2rg / sn_h2, 'b-')#, label="J Band")
#plt.plot(sn_h2, np.zeros(len(sn_h2))+np.sqrt(qe_h2rg / qe_h2) , 'b--')


#Now H band
qe_h2 = 0.75
qe_h2rg = .85
signal_h2 = qe_h2 * flux * t
signal_h2rg = qe_h2rg * flux * t

shot_h2 = np.sqrt(signal_h2)
shot_h2rg = np.sqrt(signal_h2rg)

sn_h2 = signal_h2 / np.sqrt(rn_h2**2 + d_h2*t + shot_h2**2)
sn_h2rg = signal_h2rg / np.sqrt(rn_h2rg**2 + d_h2rg*t + shot_h2rg**2)

plt.plot(sn_h2, sn_h2rg / sn_h2, 'g-')#, label="H Band")
#plt.plot(sn_h2, np.zeros(len(sn_h2))+np.sqrt(qe_h2rg / qe_h2) , 'g--')

#Now K band
qe_h2 = 0.78
qe_h2rg = 0.85
signal_h2 = qe_h2 * flux * t
signal_h2rg = qe_h2rg * flux * t

shot_h2 = np.sqrt(signal_h2)
shot_h2rg = np.sqrt(signal_h2rg)

sn_h2 = signal_h2 / np.sqrt(rn_h2**2 + d_h2*t + shot_h2**2)
sn_h2rg = signal_h2rg / np.sqrt(rn_h2rg**2 + d_h2rg*t + shot_h2rg**2)

plt.plot(sn_h2, sn_h2rg / sn_h2, 'r-')
#plt.plot(sn_h2, np.zeros(len(sn_h2))+np.sqrt(qe_h2rg / qe_h2) , 'r--')


plt.rcParams.update({'font.size':16})
plt.xlim(0,35)
plt.legend(('J Band', 'H Band', 'K Band'), loc='upper right')
plt.title("Signal To Noise Improvement of H2RG over H2 Detector")
plt.xlabel("H2 S/N Ratio")
plt.ylabel(r'(S/N)$_{\rm{H2RG}}$ / (S/N)$_{\rm{H2}}$')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot():
    phase_meas = np.array([80, 73, 73,73, 0, 74, 52, 52, 0,
                           55, 41, 95, 100, 0, 27, 18, 48, 37,
                           62, 62, 0, 24, 0, 99])
    exp_tim_meas = np.array([9.1875, 4.8828125, 5.859375, 9.375, 54,
                             4.59375, 50, 21.09375, 30, 18.75, 31.25,
                             2.4, 2, 30, 30, 56, 25.625, 23.4375, 9.8,
                             9.555, 68.90625, 29.4, 60, 2])

    phase_sim = np.array([0,10,20,30,45,50,60,70,80,90, 100])
    exp_tim_sim = np.array([60.00002948, 49.8361698, 38.5184951,
                             28.76335949, 18.38745895, 15.87178255,
                             11.85263944, 8.818400488, 6.432267964,
                             4.418926952, 1.896363903])

    phase_sim_2 = np.arange(101)
    pa = np.arccos(phase_sim_2/50.-1)
    mag = -12.73 + 1.49 * abs(pa) + 0.043 * pa**4
    flux = 2.512**(-1*mag) + 4000
    exp_tim_sim_2 = 242101 / flux

    plt.rcParams.update({'font.size':16})
    
    plt.plot(phase_meas, exp_tim_meas, 'ro', label='Measured', markersize=8)
    plt.plot(phase_sim_2, exp_tim_sim_2, 'b-', label='Predicted', linewidth=2)
    
    plt.legend(numpoints=1)#handlelength=1)
    plt.title("Moon Phase vs. Exposure Time")
    plt.xlabel("Moon Phase (%)")
    plt.ylabel("Exposure time (sec @ ISO 6400 f2.8)")
    plt.show()

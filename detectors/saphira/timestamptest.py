import numpy as np
#from scipy.stats import mode
import matplotlib.pyplot as plt
import pdb


def test():
    #times = np.loadtxt('images/saphira_10:26:28.238899839.txt', dtype='str')
    times = np.loadtxt('images/saphira_14:46:36.269917196.txt', dtype='str')
    #times = np.loadtxt('images/saphira_14:46:51.541667336.txt', dtype='str')
    #times = np.loadtxt('images/saphira_14:47:44.962387627.txt', dtype='str')

    nicetimes = np.zeros(len(times))
    timediffs = np.zeros(len(times)-1)

    fixcount = 0
    for i in range(len(times)):
        #check if seconds decimal incremented but seconds didn't
        #seconds place is times[i][6:8]
        if i < len(times)-1: #if you aren't at the end of the list
            if times[i][9:] > times[i+1][9:]:
                #print "before"
                #print "i:", i
                #print times[i-3:i+7]
                
                if times[i][6:8] == '59':
                    print "WELL CRAP, you should have coded overflow."
                    pdb.set_trace()
                    
                j = 1
                while times[i+j][6:8] <= times[i][6:8]:
                    times[i+j] = times[i+j][:6] + \
                                 str(int(times[i+j][6:8])+1) + \
                                 times[i+j][8:]
                    j += 1
                    fixcount += 1
                #print "after"
                #print times[i-3:i+7]
            
        nicetimes[i] = float(times[i][:2])*3600. + \
                       float(times[i][3:5])*60. + \
                       float(times[i][6:])

    print "This code fixed Oli's bad clock", fixcount, "times."
        
    for i in range(len(timediffs)):
        timediffs[i] = nicetimes[i+1] - nicetimes[i]

    print "Framerate was about", 1. / np.median(timediffs), "Hz."

    timediffs /= np.median(timediffs) #converts it from secs to frame numbers
    
    myhist = plt.hist(timediffs, bins = np.max(timediffs)/0.1,
                      log='True', range=[np.min(timediffs), 1.1*np.max(timediffs)])
    plt.axvline(x=np.median(timediffs), color='r')
    plt.ylim((0.5, 1e4))
    #plt.xlim((0,np.max(timediffs)+1))
    plt.show()
    #pdb.set_trace()
    
    #mymode = mode(timediffs)

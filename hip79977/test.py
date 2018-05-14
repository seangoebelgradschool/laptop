def main():
    #how freaking difficult is to to draw a box?
    #this is the simplified version of the code used in showroi.py
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb

    roirange=[100, 30, 22]
    lrmax=45

    x1 = 0.5 * roirange[0]
    x2 = -0.5 * roirange[0]
    x3 = -0.5 * roirange[0]
    x4 = 0.5 * roirange[0]
    
    y1 = 0.5 * roirange[1]
    y2 = 0.5 * roirange[1]
    y3 = -0.5 * roirange[1]
    y4 = -0.5 * roirange[1]

    v1 = [x1, y1]
    v2 = [x2, y2]
    v3 = [x3, y3]
    v4 = [x4, y4]

    R = [[ np.cos(np.deg2rad(roirange[2])) , -1*np.sin(np.deg2rad(roirange[2])) ] ,
         [ np.sin(np.deg2rad(roirange[2])) ,    np.cos(np.deg2rad(roirange[2])) ] ]

    x1,y1 = np.dot(R, v1)
    x2,y2 = np.dot(R, v2)
    x3,y3 = np.dot(R, v3)
    x4,y4 = np.dot(R, v4)

    a = np.tan(np.deg2rad(roirange[2]))
    b_t = y1 - x1 * np.tan(np.deg2rad(roirange[2]))
    b_b = y4 - x4 * np.tan(np.deg2rad(roirange[2]))

    r = lrmax
    nx1 = (-1*a*b_t + np.sqrt(-1 * b_t**2 + r**2 + a**2 * r**2) ) / (1+a**2)
    nx2 = (-1*a*b_t - np.sqrt(-1 * b_t**2 + r**2 + a**2 * r**2) ) / (1+a**2)
    nx3 = (-1*a*b_b - np.sqrt(-1 * b_b**2 + r**2 + a**2 * r**2) ) / (1+a**2)
    nx4 = (-1*a*b_b + np.sqrt(-1 * b_b**2 + r**2 + a**2 * r**2) ) / (1+a**2)

    ny1 = (b_t - np.sqrt(a**2 * r**2 - a**2 * b_t**2 + a**4 * r**2) ) / (1+a**2)
    ny2 = (b_t + np.sqrt(a**2 * r**2 - a**2 * b_t**2 + a**4 * r**2) ) / (1+a**2)
    ny3 = (b_b + np.sqrt(a**2 * r**2 - a**2 * b_b**2 + a**4 * r**2) ) / (1+a**2)
    ny4 = (b_b - np.sqrt(a**2 * r**2 - a**2 * b_b**2 + a**4 * r**2) ) / (1+a**2)

    #plot four sides of big rectangle
    plt.plot([x2, x1], [y2, y1], 'k-')
    plt.plot([x3, x2], [y3, y2], 'k-')
    plt.plot([x3, x4], [y3, y4], 'k-')
    plt.plot([x1, x4], [y1, y4], 'k-')
    
    #plot two sides of inner rectangle
    #before
    #plt.plot([nx2, nx1], [ny2, ny1], 'r-', linewidth=2)
    #plt.plot([nx3, nx4], [ny3, ny4], 'r-', linewidth=2)
    #after
    plt.plot([nx1, nx2], [ny2, ny1], 'r-', linewidth=2)
    plt.plot([nx4, nx3], [ny3, ny4], 'r-', linewidth=2)
    
    #plot the circle bounded by lrmax
    theta = np.arange(0, 2*np.pi, 0.01)
    plt.plot(lrmax * np.cos(theta) , \
             lrmax * np.sin(theta) , \
             'k-')
    plt.plot(lrmax * np.cos(theta) , \
             lrmax * np.sin(theta) , \
             'k-')

    #before
    #theta1 = np.arctan2(ny1 , nx1)
    #theta2 = np.arctan2(ny2 , nx2)
    #theta3 = np.arctan2(ny3 , nx3)
    #theta4 = np.arctan2(ny4 , nx4)
    #after
    theta1 = np.arctan2(ny1 , nx2)
    theta2 = np.arctan2(ny2 , nx1)
    theta3 = np.arctan2(ny3 , nx4)
    theta4 = np.arctan2(ny4 , nx3)

    plt.plot(lrmax * np.cos(np.linspace(theta2, theta3, 50)) , \
             lrmax * np.sin(np.linspace(theta2, theta3, 50)) , \
             'b-', linewidth=2)
    plt.plot(lrmax * np.cos(np.linspace(theta1, theta4, 50)) , \
             lrmax * np.sin(np.linspace(theta1, theta4, 50)) , \
             'b-', linewidth=2)
    
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.show()

    #pdb.set_trace()

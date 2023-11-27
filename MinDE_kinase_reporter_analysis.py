#import necessary packages 

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
import time
import math
import nd2
import scipy
from scipy import signal 

#load example image

example_U2OS_filepath = r'example_U2OS_PKAreporter.tif'
example_U2OS_image = tifffile.imread(example_U2OS_filepath)


'''
In this case, instead of using a polynomial correction, we use FIR filtering to zero-center the data and to isolate only the real signals of interest.
The conditions of the filtering may have to be adjusted depending on the acquistion parameters of your own data. 
'''

###carrier signal, should correspond to MinDE
preM=(scipy.signal.filtfilt(signal.firwin(51, [0.05,0.40],pass_zero=False),1,example_U2OS_image[:,0,:,:],axis=0))
###data signal, should correspond to the reader domain fluorescence channel
preP=(scipy.signal.filtfilt(signal.firwin(51, [0.05,0.40],pass_zero=False),1,example_U2OS_image[:,1,:,:],axis=0))

'''
A Hilbert transform is applied to the filtered signals.
'''

preM_H=scipy.signal.hilbert(preM,axis=0)
preP_H=scipy.signal.hilbert(preP,axis=0)

'''
The Hilbert transform is further processed to generate a power fraction that shows the change in reader domain signal relative to the carrier.
'''

power_fraction=((np.abs(preP_H))/(np.abs(preM_H)))*100
power_fraction_lowpass=(scipy.signal.filtfilt(signal.firwin(21, 0.005),1,power_fraction,axis=0))[25:-25]

'''
Save image power specturm to viewing in ImageJ. Each frame is a time point.
The intensity of the pixel is the power fraction at that location.

This value is normalized to the MinD intensity at that location. A high value indicates protein-protein interaction. 
'''

tifffile.imsave('power_fraction_lowpass.tif',power_fraction_lowpass)
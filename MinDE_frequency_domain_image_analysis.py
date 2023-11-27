'''
Fluorescence intensity time series data of oscillatory MinDE signals in metazoan cells can be processed using Digital Signal Processing tools. 
Remapping time domain data to the frequency domain enables a new class of image analysis.

An example image to analyze is provided. 
This example image is of two recently divided U2OS cells expressing the MinDE system. 
The image is collected at 1 FPS for 181 time points (total time: 3 minutes) using 488 and 555 fluorescence.
'''

#import necessary packages

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import time

#load example image of a pair of recently divided cells displaying oscillatory MinDE dynamics

example_U2OS_filepath = r'example_U2OS.tif'
example_U2OS_image = tifffile.imread(example_U2OS_filepath)

#split image into individual fluorescence channels for analysis

#red fluorescence channel
r = example_U2OS_image[:,1]
#green fluorescence channel
g = example_U2OS_image[:,0]

#specify sampling parameters
sampling_freq = 1 #Hz
sampling_time_points = 181 #time points



#view the first frame of the red channel
plt.figure()
plt.imshow(r[0], cmap='plasma')
plt.title('frame 0, red fluorescence channel')

#view the first frame of the green channel
plt.figure()
plt.imshow(g[0], cmap='viridis')
plt.title('frame 0, green fluorescence channel')

'''
We will use a polynomial correction function to zero center our data. 
Oscillatory data that is not zero-centered is considered to be an oscillation with a 0 Hz frequency shift away from 0. 
Therefore, frequency-based analysis will report a high power 0 Hz frequency signal corresponding to this shift.
'''

#zero center data

def polynomialCorrectImageTimeseries(img_stack):
    start_time = time.time()
    swap = img_stack.swapaxes(0,2).swapaxes(0,1)
    time_series = swap.flatten().reshape(img_stack.shape[1]*img_stack.shape[2], img_stack.shape[0])
    x = np.arange(img_stack.shape[0])
    fit = np.polynomial.polynomial.polyfit(x, time_series.T, 3) 
    interpol = np.polynomial.polynomial.polyval(x, fit)
    fitted_values = time_series-interpol
    time_series_corrected = fitted_values.flatten().reshape(img_stack.shape[1], img_stack.shape[2], img_stack.shape[0])
    corrected_img_stack = time_series_corrected.swapaxes(0,2).swapaxes(1,2)
    print("--- %s seconds ---" % (time.time() - start_time))
    return corrected_img_stack

print('Zero-center data:')
r_corr=polynomialCorrectImageTimeseries(r)
print('COMPLETE')

'''
Pixel-level analysis of oscillatory MinDE signals.
'''

print('Begin FFT analysis:')
print('calculating FFT for image...')
img_fft=np.fft.fft(r_corr,axis=0)[0:int(r_corr.shape[0]/2)]
print('calculating power and freq...')
img_power_spectra = np.abs(img_fft)
img_power=np.max(np.abs(img_fft),axis=0)
img_freq=np.argmax(np.abs(img_fft),axis=0)*sampling_freq/(sampling_time_points)
print('COMPLETE')


'''
View the power and frequency of the MinDE signals within the image.
'''

plt.figure()
plt.imshow(img_power)
plt.title('MinDE power')

plt.figure()
plt.imshow(img_freq)
plt.title('MinDE freq')


'''
Save image power specturm to viewing in ImageJ. Each frame is a frequency slice (can be converted to true frequency using the sampling parameters).
The intensity of the pixel is the power at that frequency slice at that location.
ImageJ temporal color code can be used to color cells by frequency.
'''

tifffile.imsave('img_power_spectra.tif',img_power_spectra)
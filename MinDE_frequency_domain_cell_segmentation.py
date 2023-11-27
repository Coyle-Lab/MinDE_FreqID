'''
Analyze MinDE system behavior by aggregating system behavior from thousands of individual cells.

'''

#import necessary packages

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
import nd2

#zero-center data

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

#functions for frequency domain image analysis

def generate_fstack(img_freq, img_power,img_power_cutoff=0):
    kernel=np.ones((5,5),np.uint8)
    bigK=np.ones((25,25),np.uint8)
    smallK=np.ones((3,3),np.uint8)
    fstack=list() #this will hold the frequency binned images that we contour
    for i in range(0,60):
        th=np.where(np.logical_and(img_freq==i,img_power>img_power_cutoff),255,0).astype('uint8') #create a thresholded represnetaiton for contouring
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel) # remove the small noisy parts
        closing=cv2.morphologyEx(opening, cv2.MORPH_CLOSE, bigK) # fill in any holes
        eroded=cv2.erode(closing,smallK,iterations = 3) #trim back the edges of the contours 
        fstack.append(eroded) #add the final post-processed thresholded image to the stack
    return fstack

def generate_fcontours(fstack, img_r,area_cutoff=1000):
    freq_contours=list()
    for i in range(0,60):
        freq_cells=list()
        contours,heirarchy=cv2.findContours(fstack[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            img_c=np.zeros_like(img_r[0])
            img_c=cv2.drawContours(img_c, [c], 0, 255, -1)
            points=np.where(img_c==255)
            area = cv2.contourArea(c)
            if area>area_cutoff:
                freq_cells.append([points,area,c])
            print(f"processed {i}")
        freq_contours.append(freq_cells)
    return freq_contours


'''
The following function is to analyze an image containing many cells displaying MinDE signals.
Input the image as img into the function.
The image will find each unique cell and analyze it independently.
The function will return:
    0) frequency
    1) red fluorescence intensity
    2) green fluorescence intensity
    3) power
    4) pixel locations of the cell being analyzed
    5) the number of pixels that make up that cell
    6) the area of the cell
    7) the OpenCV contour 
    
'''

def analyzeSingleImage(img, points):
    print('loading image...')

    img_r = img[:,1]
    img_g = img[:,0]
    del(img)

    print('calculating averages...')

    r_mean=np.mean(img_r,axis=0)
    g_mean=np.mean(img_g,axis=0)


    print('correcting red channel...')
    img_r=polynomialCorrectImageTimeseries(img_r)

    print('FFT analysis...')
    img_fft=np.fft.fft(img_r,axis=0)[0:int(img_r.shape[0]/2)]
    print('calculating power and freq...')
    img_power=np.max(np.abs(img_fft),axis=0)
    img_freq=np.argmax(np.abs(img_fft),axis=0)
    del(img_fft)
    
    print('generating fstack...')
    
    fstack = generate_fstack(img_freq, img_power,img_power_cutoff=800)
    
    ##now locate and store the points in the image associated with each contours in each frequency bin
    
    print('generating fcontours...')
    
    freq_contours = generate_fcontours(fstack, img_r,area_cutoff=0)
    
    for i in range(0,60):
        for c in freq_contours[i]:
            points.append([i,
                           np.median(img_freq[c[0]]),
                           np.mean(r_mean[c[0]]),
                           np.mean(g_mean[c[0]]),
                           np.mean(img_power[c[0]]),
                           c[0],
                           len(c[0]),
                           c[1],
                           c[2],
                           ])
            
    return points




'''
The following function will aggregate single cell data across many images that are all located within a single folder:

How to use:
accepts input in the form of a folder path with ND2 images
loops through the images while collecting single cell data and will return data as an array 
'''

def analyzeImageFolder(folder):
    
    print('starting analysis...')
    
    os.chdir(folder)
    
    images = [img for img in os.listdir(folder) if img.endswith(".nd2")]
    imagesLength = len(images)
    
    start_time = time.time()
    
    points=list()
    
    for i in range(imagesLength):
        
        print(f'processing location {i}')
        
        image = nd2.imread(folder + '\\' + images[i])

        points = analyzeSingleImage(image, points)
        
        print(f"finished processing location {i}   --- %s seconds ---" % (time.time() - start_time))   
        
    print("finished analyzing all images!")
    
    return np.array(points)


'''
Analyze a collection of images using the following code. 
Images must be in nd2 format. 
All images should be in one folder and the folder name should be provided as "image_folder". 
Correct the sampling_frequency and N_samples according to the way the data was collected.
'''

###EXAMPLE OF HOW TO USE THE CODE IS BELOW
###It is possible to test this code with the example_U2OS image, but note that there will only be 2 data points

'''
image_folder = #path to image folder
points = analyzeImageFolder(image_folder)

#plot every cell

#convert frequency bin to real frequency
sampling_frequency = 1 # Hz
N_samples = 181
conversion_factor = sampling_frequency/N_samples


#filter points
#1) filter for cells where the median frequency and mean frequency for all the pixels in the cells are no more different than 1 frequency bin
#2) low frequency bins are noisy, only find cells that have a frequency above frequency bin 3
#3) real oscillatory data has high power, noise will have low power, only find cells with power above the noise
correlets=np.where(np.logical_and(np.logical_and(np.abs(points[0]-points[1])<2,points[1]>3),points[7]>100))


#view analysis

plt.figure()
plt.scatter(points[3][correlets]/points[2][correlets],points[1][correlets]*conversion_factor,c=points[1][correlets],c='gray', label='name here',alpha=0.1)
plt.set_xlabel('[MinE]/[MinD]')
plt.set_ylabel('Frequency (Hz)')



#function to prepare simplified representation of the aggregate data

def simplePhasePortrait(x,y):
    points = list()
    for i in range(int(np.min(y)), int(np.max(y)+1)):
        values = x[np.where(y==i)]
        if values.size != 0:
            mean = np.mean(values)
            std = np.std(values)
            point = [i, mean, std]
            points.append(point)
        else:
            pass
    return np.array(points).T

plt,figure()
simple_phase_portrait = simplePhasePortrait(points[3][correlets]/points[2][correlets],points[1][correlets])
plt.errorbar(simple_phase_portrait[1], simple_phase_portrait[0]*conversion_factor, xerr=simple_phase_portrait[2], fmt = 'o')
plt.set_xlabel('[MinE]/[MinD]')
plt.set_ylabel('Frequency (Hz)')

'''
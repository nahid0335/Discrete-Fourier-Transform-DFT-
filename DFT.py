# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:45:56 2021

@author: User
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def HighPassFilter(img,radius):
    # Circular HPF mask, center circle is 0, remaining all ones
    #Can be used for edge detection because low frequencies at center are blocked
    #and only high frequencies are allowed. Edges are high frequency components.
    #Amplifies noise.
    
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    
    mask = np.ones((rows, cols, 2), np.uint8)
    r = radius                                     #80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    
    '''
    mask = np.ones((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0
    '''
    
    return mask


def LowPassFilter(img,radius):
    # Circular LPF mask, center circle is 1, remaining all zeros
    # Only allows low frequency components - smooth regions
    #Can smooth out noise but blurs edges.
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = radius                                          #100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    
    return mask


def BandPassFilter(img,innerRadius,outerRadius):
    # Band Pass Filter - Concentric circle mask, only the points living in concentric circle are ones
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r_out = outerRadius                                       #80
    r_in = innerRadius                                        #10
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    
    return mask



img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('input image',img)

#create dft from input image
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

#Without shifting the data would be centered around origin at the top left
#Shifting it moves the origin to the center of the image. 
dft_shift = np.fft.fftshift(dft)

#Calculate magnitude spectrum from the DFT (Real part and imaginary part)
magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

#fourier spectrum
img1 =cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('magnitude_spectrum',img1)

'''
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,2)
ax1.imshow(img)
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,1)
ax2.imshow(dft_shift)
ax2.title.set_text('FFT of image')
plt.show()


phaseimage = np.log(cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1]))
img2 =cv2.normalize(phaseimage, phaseimage, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('phase image',img2)
'''

# apply mask and inverse DFT: Multiply fourier transformed image (values)
#with the mask values. 

#edge detection
mask1 = HighPassFilter(img, 80)


fshift1 = dft_shift * mask1

f_ishift1 = np.fft.ifftshift(fshift1)
img_back1 = cv2.idft(f_ishift1)


edgeimage = cv2.magnitude(img_back1[:,:,0],img_back1[:,:,1])
img11 =cv2.normalize(edgeimage, edgeimage, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('Edge detection(HPF)',img11)

'''
ph1 = cv2.phase(img_back[:,:,0],img_back[:,:,1])
img12 =cv2.normalize(ph1, ph1, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('ph1',img12)
'''

#smoothing
mask2 = LowPassFilter(img, 100)


fshift2 = dft_shift * mask2

f_ishift2 = np.fft.ifftshift(fshift2)
img_back2 = cv2.idft(f_ishift2)


smoothimage = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])
img2 =cv2.normalize(smoothimage, smoothimage, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('Smoothing(LPF)',img2)



#BPF
mask3 = BandPassFilter(img, 10, 80)


fshift3 = dft_shift * mask3

f_ishift3 = np.fft.ifftshift(fshift3)
img_back3 = cv2.idft(f_ishift3)


bpfimage = cv2.magnitude(img_back3[:,:,0],img_back3[:,:,1])
img3 =cv2.normalize(bpfimage, bpfimage, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('Band Pass Filter',img3)


cv2.waitKey(0)
cv2.destroyAllWindows()
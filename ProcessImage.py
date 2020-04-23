import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os, glob
import random
import pickle

import ProcessingPipeline

try:
    from cv2 import cv2
except:
    import cv2


class ProcessImage:
    def __init__(self):
        # print('Processing Image...')
        pass

    def _sharpen_image(self, image):
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        return sharpened

    def _compute_sobel_bin(self, image, sobel_kernel=3, thresh=(10, 100)):
        '''
        Compute gradients using sobel filter
        :param image: ndarray
        :param sobel_kernel: int
        :param thresh: tuple[2]
        :return: ndarray dim >= 2
        '''
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        lxbinary = np.zeros_like(scaled_sobelx)
        lxbinary[(scaled_sobelx >= thresh[0])
                 & (scaled_sobelx <= thresh[1])] = 1
        return lxbinary

    def pipeline(self, img, s_thresh=(150, 255), sx_thresh=(20, 100),
                 R_thresh=(200, 255), G_thresh=(200, 255), sobel_kernel=3):
        '''
        Pipeline to create binary image.
        Gradients in l and s channel of HSV
        Good bright pixel from R channels for Yellow and white
        Create color mask in HSV for white and Yellow
        Compute Logical AND between gradients images of s channel l channel and R bright.
        Compute logical and between color mask and processed grads combined
        This is the best threshold for images
        :return images: nd.array
        '''
        dst = np.copy(img)
        red_channel = dst[:, :, 0]
        hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        lxbinary = self._compute_sobel_bin(l_channel, sobel_kernel)
        s_binary = np.zeros_like(s_channel, dtype=np.uint8)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        s_binary = self._compute_sobel_bin(s_channel)

        R_binary = np.zeros_like(red_channel)
        R_binary[(red_channel >= R_thresh[0]) & (red_channel <= R_thresh[1])] = 1
        R_binary = cv2.medianBlur(R_binary, 5)
        s_binary = cv2.medianBlur(s_binary, 5)
        lxbinary = cv2.medianBlur(lxbinary, 15)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(hsv, (16, 70, 100), (36, 255, 255))
        white_1 = cv2.inRange(hsv, (0, 0, 200), (255, 20, 255))
        white_2 = cv2.inRange(hls, (0, 200, 0), (255, 255, 200))
        white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))
        color_mask = yellow | white_1 | white_2 | white_3
        # plt.imshow(yellow, cmap='gray')
        # plt.show()
        # plt.imshow(white_1, cmap='gray')
        # plt.show()
        # plt.imshow(white_2, cmap='gray')
        # plt.show()
        # plt.imshow(white_3, cmap='gray')
        # plt.show()
        k = np.ones((61, 61))
        boximg = np.zeros_like(color_mask)
        cmMod = cv2.boxFilter(color_mask, 0, (15, 15), boximg, (-1, -1), False, cv2.BORDER_DEFAULT)
        # plt.imshow(boximg, cmap='gray')
        # plt.show()
        # plt.imshow(cmMod, cmap='gray')
        # plt.show()

        combined_binary = np.zeros_like(lxbinary)
        # combined_binary[((s_binary == 1) & (R_binary == 1)) | ((lxbinary == 1) & (R_binary == 1)) | ((lxbinary == 1) & (s_binary == 1))] = 1
        combined_binary[((((R_binary == 1) & (s_binary == 1)) & (lxbinary == 1)))] = 1
        # temp = np.logical_or(boximg, combined_binary)
        # combined_binary = combined_binary.astype(int)
        # combined_binary[boximg | combined_binary] = 1
        combined_binary = cv2.bitwise_or(combined_binary, boximg)
        # cv2.rectangle(combined_binary, (20, 20), (200, 200), (255, 255, 255), 2)
        # plt.imshow(combined_binary)
        # plt.show()
        return combined_binary

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(20, 100)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            abs_sobel = np.absolute(sobelx)
        if orient == 'y':
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            abs_sobel = np.absolute(sobely)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output


if __name__ == '__main__':
    # img = plt.imread('test_images/test_hard3.jpg')
    img = plt.imread('test_images/test5.jpg')
    proc = ProcessingPipeline.ProcessingPipeline()
    op = proc.get_undist_img_test(img)
    imgProc = ProcessImage()
    op2 = imgProc.pipeline(op)
    plt.imshow(op)
    plt.show()
    plt.imshow(op2)
    plt.show()
    # proc.plot_side_by_side(op2, op)

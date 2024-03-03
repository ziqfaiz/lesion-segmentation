# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np

input_dir = 'dataset/test'
output_dir = 'dataset/output'

def segmentImage(image):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
   
    # Binarize input image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _,binarized_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_OTSU)    

    #Define boundary rectangle containing the foreground object
    height, width = binarized_image.shape
    left_margin_proportion = 0.2
    right_margin_proportion = 0.2
    up_margin_proportion = 0.1
    down_margin_proportion = 0.1
    
    boundary_rectangle = (
        int(width * left_margin_proportion),
        int(height * up_margin_proportion),
        int(width * (1 - right_margin_proportion)),
        int(height * (1 - down_margin_proportion)),
    )
    
    number_of_iterations = 10

    # Initialize the mask with known information
    mask = np.zeros((height, width), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    mask[binarized_image == 0] = cv2.GC_FGD
    #mask = np.zeros((height, width), np.uint8)

    # Arrays used by the algorithm internally
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(
        image,
        mask,
        boundary_rectangle,
        background_model,
        foreground_model,
        number_of_iterations,
        cv2.GC_INIT_WITH_MASK,
    )

    grabcut_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(
    "uint8"
    )
    segmented_image = image.copy() * grabcut_mask[:, :, np.newaxis]

    new = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    
    _,outImg = cv2.threshold(new, 128, 255, cv2.THRESH_OTSU)

    #outImg = cv2.cvtColor(outImg, cv2.COLOR_GRAY2BGR)
    #outImg = cv2.bitwise_not(outImg)
    
    
    # END OF YOUR CODE
    #########################################################################
    #return cv2.bitwise_not(binarized_image)
    return outImg

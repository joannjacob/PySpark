import cv2
import numpy as np


def roi(image):

    # Select ROI
    r = cv2.selectROI(image, False)

    # Crop image
    imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
    return imCrop

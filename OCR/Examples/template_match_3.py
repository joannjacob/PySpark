import imutils
import cv2
import numpy as np
from resize_image import image_resize


img_rgb = cv2.imread(
    '/home/qburst/OCR/Images/instrument_images/Files/IMG_2487_rotated.JPG')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('./templates/template1_blank.jpg', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

cv2.imshow('Detected', image_resize(img_rgb, height=800))
cv2.waitKey(0)

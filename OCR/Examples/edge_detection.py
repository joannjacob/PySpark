import cv2
import numpy as np
from resize_image import image_resize

image = cv2.imread(
    '/home/qburst/xwa_poc_backend/images/instrument_one/IMG_2493_rotated.JPG')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Original', image_resize(image, height=800))
edges = cv2.Canny(image, 100, 200)
cv2.imshow('Edges', image_resize(edges, height=800))
cv2.waitKey(0)

# k = cv2.waitKey(5) & 0xFF
# if k == 27:
#     break


cv2.destroyAllWindows()


import numpy as np
import cv2
import imutils
from resize_image import image_resize

template = cv2.imread('./templates/template1_generic.jpg')  # template image
image_o = cv2.imread(
    '/home/qburst/xwa_poc_backend/images/instrument_one/IMG_2490_rotated.JPG')  # image

template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.threshold(template, 190, 255, cv2.THRESH_BINARY)
# ret, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
# template = cv2.Canny(template, 50, 200)

image = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)
visualize = True


# Store width and height of template in w and h
w, h = template.shape[::-1]
found = None

for scale in np.linspace(0.2, 1.0, 20)[::-1]:

    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(image_o, width=int(image.shape[1] * scale))
    r = image.shape[1] / float(resized.shape[1])

    # if the resized image is smaller than the template, then break
    # from the loop
    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image edged
    # = cv2.Canny(resized, 50, 200) result = cv2.matchTemplate(edged, template,
    # cv2.TM_CCOEFF) (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    # if we have found a new maximum correlation value, then update
    # the found variable if found is None or maxVal > found[0]:
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # ret, edged = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    # edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    print(maxVal, maxLoc)
    # check to see if the iteration should be visualized
    if visualize:
        # draw a bounding box around the detected region
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
        cv2.imshow("Visualize", image_resize(clone, height=800))
        cv2.waitKey(0)

    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

# unpack the found varaible and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))

# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image_resize(image, height=800))
cv2.waitKey(0)

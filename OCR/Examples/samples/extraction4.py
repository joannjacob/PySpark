import matplotlib.pylab as plt
from PIL import Image
from ast import literal_eval
import pytesseract
import csv
import re
import os
import cv2
from common import preprocess_image, ocr_image
from roi import roi
import numpy as np


def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop


def process_instrument_one(file_name):
    image = cv2.imread(file_name)

    image = preprocess_image(image, True, True, False)

    # find contours / rectangle
    # contours = cv2.findContours(image, 1, 1)[0]
    # rect = cv2.minAreaRect(contours[0])

    # crop
    # img_croped = crop_minAreaRect(image, rect)
    cv2.imshow('cropped region', image)
    cv2.waitKey(0)

    image = roi(image)

    text = ocr_image(image)
    return text


result = process_instrument_one('table5.JPG')

# result = pytesseract.image_to_string(Image.open('table1.jpg'), lang="eng")

# print(type(result))
print(result)

with open('result.txt', 'w') as outfile:
    outfile.write(result)

    # with open('people.csv', 'w') as outfile:
    #     writer = csv.writer(outfile)
    # writer.replace(",", "")
    # writer.writerow(result)

    # string = open('people.csv').read()
    # new_str = re.sub('[^a-zA-Z0-9\n\.]', ' ', string)
    # # print(string, "/n")
    # open('people.csv', 'w').write(new_str)

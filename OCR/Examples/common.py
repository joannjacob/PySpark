import cv2
import math
from pytesseract import *
from resize_image import image_resize
import numpy as np
from scipy.ndimage import interpolation as inter

# Increase brightness


def increase_brightness(image, value=50):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v < lim] = 0
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image


# Remove noise
def remove_noise(image):
    blur_img = cv2.GaussianBlur(image, (5, 5), 0)
    return blur_img


def rotate_image(image):

    image_edges = cv2.Canny(image, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(image_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    angle_deg = np.median(angles)
    print(angle_deg)
    if angle_deg < 0:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        angle_deg += 90
    elif angle_deg > 0:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        angle_deg -= 90

    return image


def perspective_transform(image):
    rows, cols = image.shape
    pts1 = np.float32(
        [[cols*.25, rows*.95],
         [cols*.90, rows*.95],
         [cols*.10, 0],
         [cols,     0]]
    )

    pts2 = np.float32(
        [[cols*0.1, rows],
         [cols,     rows],
         [0,        0],
         [cols,     0]]
    )

    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (cols, rows))
    return image


def crop_min_area_rect(img, rect):

    # rotate img
    angle = rect[2]

    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]
    return img_crop


def correct_skew(thresh, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated


def preprocess_image(image, grey_scale, resize, extract_rect):
    # print(image.shape)
    if resize:
        image = image_resize(image, height=800)
    # print(image.shape)

    # image = increase_brightness(image, 50)

    if grey_scale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = perspective_transform(image)

    # image = remove_noise(image)

    # Perform rotation only if image is not taken straight. Rotation is applied for the sample image to make it correct.
    # image = rotate_image(image)
    # image = correct_skew(image)

    if extract_rect:
        thresh = cv2.threshold(image, 150, 255, 0)[1]
        contours = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        rects = []
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                rects.append(approx)
        c = max(rects, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        image = crop_min_area_rect(image, rect)
        cv2.imshow('cropped region', image)

    # kernel = np.ones((1, 1), np.uint8)
    # processed_image = cv2.erode(image, kernel, iterations=1)

    return image


def ocr_image(image):
    text = pytesseract.image_to_string(
        image, lang='eng', config='--oem 1 --psm 6')
    return text


def preprocess_template(image, grey_scale, resize):
    # print(image.shape)
    if resize:
        image = image_resize(image, height=250)

    # print(image.shape)

    # image = increase_brightness(image, 50)

    if grey_scale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = remove_noise(image)

    # kernel = np.ones((1, 1), np.uint8)
    # processed_image = cv2.erode(image, kernel, iterations=1)

    return image

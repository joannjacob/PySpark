import cv2
import numpy as np
from resize_image import image_resize

original_img = cv2.pyrDown(cv2.imread(
    "/home/qburst/xwa_poc_backend/images/instrument_one/IMG_2490.JPG", cv2.IMREAD_UNCHANGED))
template = cv2.imread('Examples/templates/template1_generic.jpg')

img = original_img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3, 3), dtype=np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
edged = cv2.Canny(gray, 30, 200)
contours, hierarchy = cv2.findContours(
    edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

max_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(max_contour)
max_roi = original_img[y:y+h, x:x+w]
cv2.imshow('max_roi', max_roi)
img = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
cv2.imshow('contour img', image_resize(img, height=800))
cv2.waitKey(0)


# idx = 0
# max_area = cv2.contourArea(max_contour)
# max_perimeter = cv2.arcLength(max_contour, True)
# print("max", max_perimeter, max_area)

# perimeter_list = []
# area_list = []
# index = 0
# roi_contour = max_contour
# for ctr in contours:
#     a = cv2.contourArea(ctr)
#     per = cv2.arcLength(ctr, True)

#     if per > 1000 and per < max_perimeter:
#         # min_contour = ctr
#         perimeter_list.append(per)
#         area_list.append(a)
#         roi_contour = ctr
#         index = idx
#     idx += 1

# sorted_perimeter_list = sorted(perimeter_list)
# sorted_area_list = sorted(area_list)
# print(sorted_perimeter_list)
# print(sorted_area_list)

# x, y, w, h = cv2.boundingRect(roi_contour)
# roi = original_img[y:y+h, x:x+w]
# cv2.imwrite('./ROI/roi' + str(idx) + '.jpg', roi)

# cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 0), 2)
# img = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

# cv2.imshow('contour img', image_resize(img, height=800))
# cv2.imshow('img', roi)

# cv2.waitKey(0)




#####INSTRUMENT TWO ####



# idx = 0

        # max_area = cv2.contourArea(max_contour)
        # max_perimeter = cv2.arcLength(max_contour, True)
        # print("max", max_perimeter)

        # perimeter_list = []
        # area_list = []
        # index = 0
        # roi_contour = max_contour
        # for ctr in contours:
        #     a = cv2.contourArea(ctr)
        #     per = cv2.arcLength(ctr, True)
        #     perimeter_list.append(per)
        #     area_list.append(a)
        #     if per == 2257.6193997859955:
        #         # min_contour = ctr
        #         max_area = a
        #         perimeter_list.append(per)
        #         area_list.append(a)
        #         roi_contour = ctr
        #         index = idx
        #     idx += 1

        # sorted_perimeter_list = sorted(perimeter_list)
        # sorted_area_list = sorted(area_list)
        # print(sorted_perimeter_list)
        # # print(sorted_area_list)
        # print("MAXIMUM", max(sorted_area_list), max(sorted_perimeter_list))

        # x, y, w, h = cv2.boundingRect(roi_contour)
        # roi = original_image[y:y+h, x:x+w]
        # cv2.imwrite('./ROI/roi' + str(idx) + '.jpg', roi)
        # cv2.imshow("roi", roi)

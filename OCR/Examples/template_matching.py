import cv2
import numpy as np
from common import preprocess_image, preprocess_template, ocr_image
from resize_image import image_resize

try:
    # Read image and template
    img = cv2.imread(
        '/home/qburst/OCR/Images/instrument_images/Files/IMG_2489_rotated.JPG')
    # img = cv2.imread(
    #     '/home/qburst/OCR/Images/screens/IMG_0412.JPG')

    template = cv2.imread('./templates/template1_generic.jpg')

    # Perform basic image processing techniques on both image and template
    img = preprocess_image(img, True, True, False)
    template = preprocess_template(template, True, True)
    # img = image_resize(img, width=template.shape[1])
    # template = image_resize(template, height=img.shape[0])

    # cv2.imshow("input image", img)
    # cv2.imshow("template", template)
    cv2.waitKey(0)


except IOError as e:
    print("({})".format(e))
else:
    img2 = img.copy()
    w, h = template.shape[::-1]


methods = ['cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # if img.shape[0] < h:
    #     print("height less")
    #     print("template shape", template.shape, template.shape[0])
    #     print("image shape", img.shape)
    #     # print(_img.size().height)
    #     template = image_resize(
    #         template, width=template.shape[1], height=img.shape[0])

    # if img.shape[1] < w:
    #     print("width less")
    #     print("template shape", template.shape)
    #     print("image shape", img.shape)
    #     template = image_resize(
    #         template, width=img.shape[1], height=template.shape[0])
    #     # img = image_resize(img, width=w, height=img.shape[0])

    # Apply template Matching
    print("template shape", template.shape, template.size)
    print("image shape", img.shape, img.size)
    # cv2.imshow("image", img)
    # cv2.imshow("template", template)

    # cv2.waitKey(0)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # print("Method: %s", meth)
    # print("min_val: ", min_val)
    # print("max_val: ", max_val)
    # print("min_loc: ", min_loc)
    # print("max_loc: ", max_loc)
    # print(" ")

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    img = img[top_left[1]:top_left[1]+h, bottom_right[0]-w:bottom_right[0]]
    # img = image_resize(img, height=800)

    cv2.imwrite("output.jpg", img)
    cv2.waitKey(0)

# Perform all the
output_img = cv2.imread('./output.jpg')
cv2.imshow("cropped", output_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

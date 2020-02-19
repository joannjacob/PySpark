import cv2
import numpy as np
import pytesseract
from scipy.ndimage import interpolation as inter
import pytesseract


def automatic_brightness_and_contrast(image, clip_hist_percent=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def simpleThreshold(image):
    ret, thresh1 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return thresh1


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


def medianBlur(image):
    return cv2.medianBlur(image, 3)


def adaptiveThreshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def diliation(image):
    return cv2.dilate(image, kernel, iterations=1)


kernel = np.ones((5, 5), np.uint8)
image = cv2.imread("./output.jpg")

# res = automatic_brightness_and_contrast(image)
# cv2.imshow("Bright", res)

res = simpleThreshold(image)
cv2.imshow("Adaptive Tresh", res)

res = correct_skew(res)
cv2.imshow("skew", res)


cv2.imshow("original", image)

text = pytesseract.image_to_string(res, lang="eng", config='--oem 1 --psm 6')
print(text)

cv2.waitKey(0)

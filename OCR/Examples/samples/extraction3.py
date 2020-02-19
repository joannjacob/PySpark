from PIL import Image
import PIL.Image

from pytesseract import image_to_string
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = '/home/qburst/OCR/Examples/'
# TESSDATA_PREFIX = '/home/qburst/OCR/Examples/'
output = pytesseract.image_to_string(
    PIL.Image.open('IMG_0380.jpg').convert("RGB"), lang='eng')
print(output)

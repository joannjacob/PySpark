# from tesserocr import PyTessBaseAPI
# api = PyTessBaseAPI(path='/home/qburst/OCR/tesserocr-master/tessdata/', lang='eng')
# try:
#     # api.SetImageFile('digits1.jpg')
#     print(api.GetUTF8Text())
#     print(api.AllWordConfidences())
#     print(api.GetAvailableLanguages())
# finally:
#     api.End()


from tesserocr import PyTessBaseAPI

images = ['digits1.jpeg']

with PyTessBaseAPI(path='/home/qburst/OCR/tesserocr-master/tessdata/.', lang='eng+chi_sim') as api:
    for img in images:
        api.SetImageFile(img)
        print(api.GetUTF8Text())
        print(api.AllWordConfidences())

from PIL import Image
import sys
column = Image.open(sys.argv[1])
gray = column.convert('L')
blackwhite = gray.point(lambda x: 0 if x < 200 else 255, '1')
blackwhite.save("code_bw.jpg")
import cv2
import srwrapper

filename = input('Enter a filename ')
ratio = input('Enter an upsampling ratio: 2 or 4')
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
GPupsamp(img_in, ratio)
cv2.imwrite('img_out.jpg', img_out)

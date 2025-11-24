from utils import *
import cv2

img = cv2.imread('case1/page_16.jpeg')

grayImg = grayScale(img)
blurredImg = gaussianBlur(grayImg, (5, 5), 0)
cv2.imwrite('blurr.jpg', blurredImg)
otsuImg = otsu(blurredImg)
closingImg = closing(otsuImg, (3, 3))

cv2.imwrite('1_3x3_closing.jpg', closingImg)
closingImg = closing(otsuImg, (5, 5))
cv2.imwrite('1_5x5_closing.jpg', closingImg)

closingImg = closing(otsuImg, (7, 7))
cv2.imwrite('1_7x7_closing.jpg', closingImg)

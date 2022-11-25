import cv2
import numpy as np


NmFile = "../assets/Rubik.jpg"
Im = cv2.imread(NmFile, 1)
hsv = cv2.cvtColor(Im, cv2.COLOR_BGR2HSV)


HueMin = 110
HueMax = 130
SaturasiMin = 0
SaturasiMax = 255
ValueMin = 0
Valuemax = 255

BatasWarnaBawah = np.array([HueMin, SaturasiMin, ValueMin])
BatasWarnaAtas = np.array([HueMax, SaturasiMax, Valuemax])

# Treshold
mask = cv2.inRange(hsv, BatasWarnaBawah, BatasWarnaAtas)

cv2.imshow('Image', Im)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()

exit()

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

Im = cv2.imread('../assets/Object.jpg')
ImO = copy.deepcopy(Im)
# Hue Range

hsv = cv2.cvtColor(Im, cv2.COLOR_BGR2HSV)

# Nilai threshold
HueRange = (0, 360)
SaturationRange = (0, 0)
ValueRange = (100, 255)
(HMin, HMax) = HueRange
HMin = int(HMin/2)
HMax = int(HMax/2)
(SMin, SMax) = SaturationRange
(VMin, VMax) = ValueRange
LColor = np.array([HMin, SMin, VMin])
UColor = np.array([HMax, SMax, VMax])
ImBw = cv2.inRange(hsv, LColor, UColor)

# Mengaplikasikan cv2.connectedComponents()
JumlahLabel, label = cv2.connectedComponents(ImBw)

# Untuk kotak dan label
color = (255, 255, 255)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(1, JumlahLabel):
    [bb, cc] = np.where(label == i)

    # Mencari titik untuk membuat kotak
    bmax = (np.max(bb))
    bmin = (np.min(bb))
    cmax = (np.max(cc))
    cmin = (np.min(cc))
    p1 = (cmin, bmin)
    p2 = (cmax, bmax)

    #
    if bmax-bmin > 30:
        Im = cv2.rectangle(Im, p1, p2, color, thickness)
        Im = cv2.putText(Im, str(i), p1, font, 1,
                         color, thickness, cv2.LINE_AA)

cv2.imshow("Citra Treshold", cv2.cvtColor(ImO, cv2.COLOR_BGR2RGB))
cv2.imshow("Citra Dengan Label", cv2.cvtColor(Im, cv2.COLOR_BGR2RGB))
cv2.waitKey()
cv2.destroyAllWindows()

exit()

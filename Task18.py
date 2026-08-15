import cv2
import numpy as np
import matplotlib.pyplot as plt

COL_BOX = (0, 255, 0)
COL_CENTROID = (0, 0, 255)
COL_OVERLAY = (0, 255, 255)

img = np.zeros((700, 1200, 3), dtype=np.uint8)

cv2.rectangle(img, (50, 100), (250, 250), COL_BOX, 3)
cv2.circle(img, (400, 175), 75, COL_CENTROID, -1)
cv2.ellipse(img, (750, 175), (150, 70), 0, 0, 360, COL_BOX, 3)
cv2.line(img, (50, 350), (450, 350), COL_OVERLAY, 5)

pts = np.array([[550, 450], [700, 400], [850, 450], [800, 600], [600, 600]], np.int32)
cv2.polylines(img, [pts], True, COL_OVERLAY, 3)
cv2.fillPoly(img, [pts], COL_CENTROID)

cv2.arrowedLine(img, (250, 175), (325, 175), COL_CENTROID, 5, tipLength=0.2)

text = "OpenCV Drawing"
x = 450
y = 50
padding = 5

(font_width, font_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

cv2.rectangle(img, (x - padding, y - font_height - padding), (x + font_width + padding, y + baseline + padding), COL_BOX, 2)
cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

overlay = img.copy()

cv2.rectangle(overlay, (900, 450), (1100, 600), COL_OVERLAY, -1)
img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
cv2.rectangle(img, (900, 450), (1100, 600), COL_BOX, 3)

cv2.imshow("OpenCV Drawing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()






















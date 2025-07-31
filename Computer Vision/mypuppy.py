import numpy as np
import cv2

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x,y), 100, (0,255,0) -1)

cv2.namedWindow(winname = 'mydrawing')

cv2.setMouseCallback('my_drawing', draw_circle)


image = np.zeros((512, 512,3), np.int8)

while True:
    cv2.imshow("my_drawing", image)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

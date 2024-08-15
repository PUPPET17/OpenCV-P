import cv2
import numpy as np
from mss import mss

screen_region = {"top": 0, "left": 0, "width": 1920, "height": 1080}

sct = mss()
cv2.namedWindow('Screen Capture', cv2.WINDOW_NORMAL)

while True:
    screenshot = sct.grab(screen_region)
    
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    cv2.imshow("Screen Capture", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

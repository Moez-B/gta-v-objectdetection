import cv2 as cv
from windowcapture import GameCapture

#enter the name of the window here
window_name = ""

frames = GameCapture(window_name)

while True:

    frame = frames.get_frame()
    cv.imshow("screen", frame)

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

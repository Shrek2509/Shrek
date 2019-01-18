import cv2

def Binyr(r):
    retval, r = cv2.threshold(r, 117, 255, cv2.THRESH_BINARY)


    return r
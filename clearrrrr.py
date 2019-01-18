import cv2

def Erode(ll):
    po = cv2.erode(ll, None, iterations=2)
    cv2.imshow('poisk', po)
    return po
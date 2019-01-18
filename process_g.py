import cv2

def Gaus(a):
    c = cv2.GaussianBlur(a, (5, 5), 2)

    cv2.imshow("1", c)
    return c
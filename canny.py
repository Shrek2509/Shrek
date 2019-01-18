import cv2
def Canny(q):
    q = cv2.Canny(q, 0, 100)
    cv2.imshow("3", q)
    return q
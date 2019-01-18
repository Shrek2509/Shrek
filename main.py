import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab
import os
import process_g
import binar
import  canny
import clearrrrr

cv2.namedWindow("1")
cv2.namedWindow("2")
#cv2.namedWindow("3")

cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    r,g,b=cv2.split(frame)
    r = process_g.Gaus(r)
    g = process_g.Gaus(g)
    b = process_g.Gaus(b)
    i_bgr=cv2.merge((r,g,b))
    # cv2.imwrite("can.png", frame)
    # cap.release()
    gray = cv2.cvtColor(i_bgr , cv2.COLOR_BGR2HSV)


    h, s, v = cv2.split(gray)

    h = process_g.Gaus(h)
    s = process_g.Gaus(s)
    v = process_g.Gaus(v)

    fr=cv2.merge((h,s,v))

    a1=0
    a2=100
    a3=100
    k=155
    l=10

   # poisk=cv2.inRange(fr,(a1-l,a2-k,a3-k),(a1+l,a2+k,a3+k))
    # зелёный
    cv2.imshow("norm", frame)
    # poisk = cv2.inRange(fr, (36,25,25), (86,255,255))
    # mask1 = cv2.inRange(fr, (7,40,60), (18,255,200))
    mask1 = cv2.inRange(fr, (36,25,25), (86,255,255))


    # cv2.imshow("1",poisk)

    # mask = cv2.erode(poisk, None, iterations=2)
    # mask = cv2.dilate(poisk,(15,15),iterations=5)
    # cv2.imshow("2", mask)
    # Нахождение синего диапозона цвета
    mask2 = cv2.inRange(fr, (110, 50, 50), (130, 255, 255))
    # Объеденение двух масок цветов
    mask = cv2.bitwise_or(mask1, mask2)

    # Фильтр для очистки изображения
    poisk = clearrrrr.Erode(mask)

    # Нахождение радиуса Зеленого цвета
    __, thresh = cv2.threshold(poisk, 127, 255, 0)

    # Задаем переменные
    contours_area = []
    front = cv2.FONT_HERSHEY_DUPLEX

    # Поиск контуров нашего диапозона
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    count_contours = 0

    for c in contours:
        # Перебираем все найденные контуры в цикле
        x, y, w, h = cv2.boundingRect(c)

        # Вписыват в наш диапозон замкнутый контур прямоугольника
        rect = cv2.minAreaRect(c)
        # Находит четыре вершины повернутого прямоугольника
        box1 = cv2.boxPoints(rect)
        # округление координат
        box = np.int0(box1)

        # Находим площадь внутри контура
        area2 = cv2.contourArea(c)
        if 350 < area2 < 20000:
            # Рисует простой, толстый или заполненный прямоугольник справа вверх.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Наименует объекты на захваченном изображении
            cv2.putText(frame, str(round(area2)), (x, y), front, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
            print(area2)
            count_contours += 1
            # вычисляем площадь и отсекаем контуры с маленькой площадью
            area = int(rect[1][0] * rect[1][1])
    if area > 350:
        # По координаты нашего диапозона накладываем контур
        cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
        # Счетчик, на выводе экрана подсчитывает объекты
        str_count = "Count: {}".format(count_contours)
        # Наименует объекты на захваченном изображении
        cv2.putText(frame, str_count, (10, 25), front, 0.5, (0, 255, 0), lineType=cv2.LINE_AA)

        # Изменение размера изображения
        frame = cv2.resize(frame, (1280, 960))
        cv2.imshow("img2", frame)

    # waitKey отображение кадра в мили седкундах
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

    # output = cv2.bitwise_and(frame, i_bgr, mask=poisk)
    #
    # cv2.imshow("3", output)
    #
    # i_bin=binar.Binyr(mask)
    #
    # cv2.imshow("binar",i_bin)
    #
    # contours, hierarchy = cv2.findContours(i_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #
    #     x, y, w, h1 = cv2.boundingRect(c)
    #     # draw a green rectangle to visualize the bounding rect
    #     cv2.rectangle(poisk, (x, y), (x + w, y + h1), (0, 255, 0), 2)
    #
    #     # get the min area rect
    #     rect = cv2.minAreaRect(c)
    #     box = cv2.boxPoints(rect)
    #     # convert all coordinates floating point values to int
    #     box = np.int0(box)
    #     # draw a red 'nghien' rectangle
    #     cv2.drawContours(mask, [box], 0, (0, 0, 255))
        # finally, get the min enclosing circle
        # (x, y), radius = cv2.minEnclosingCircle(c)
        # # convert all values to int
        # center = (int(x), int(y))
        # radius = int(radius)
        # # and draw the circle in blue
        # cvcv = cv2.circle(frame, center, radius, (255, 0, 0), 2)
        #
        #
        # cv2.drawContours(cvcv, contours, -1, (255, 255, 0), 1)
    #
    # cv2.imshow("contours",cvcv)
    # key=cv2.waitKey(1)
    # if key==27 or key == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


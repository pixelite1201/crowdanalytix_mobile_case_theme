import numpy as np
import cv2
from matplotlib import pyplot as plt
import os                  #
TRAIN_DIR = '/home/priyanka/Test'  #  dealing with directories

for theme in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, theme)
        for j in os.listdir(path):
            path_new = os.path.join(path,j)
            img = cv2.imread(path_new)
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img2 = 255 - im_gray
            _, thresh = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, 2)
            area = []
            for i in range(len(contours)):
                area.append(cv2.contourArea(contours[i]))
            index = area.index(max(area))
            cnt = contours[index]
            x, y, w, h = cv2.boundingRect(cnt)
            crop = img[y:y + h, x:x + w]
            cv2.imwrite(path_new, crop)





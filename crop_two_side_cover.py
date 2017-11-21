# This will help in manually segmenting the mobile front and back as separate images.User will click in the middle
# and will tell whether he wants to save the left image i.e. 1 or the right i.e. 2
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os                  #
TRAIN_DIR = './Train'  #  dealing with directories
old_X = 0
old_Y = 0
def store_point(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global X
        global Y
        X,Y=x,y


for theme in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, theme)
        for j in os.listdir(path):
            path_new = os.path.join(path,j)
            print(path_new)
            img = cv2.imread(path_new,0)
            img2 = cv2.imread(path_new)
            x,y=img.shape
            if(float(x)/y < 1.5 ):
                print(float(x)/y)
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', store_point)
                cv2.imshow('image',img)
                cv2.waitKey(0)
                if(old_X is not X and old_Y is not Y):
                    old_X=X
                    old_Y=Y
                    var = raw_input("2/1/0")
                    if (int(var) is 2):
                        new_img = img2[:, X:]
                        cv2.imwrite(path_new, new_img)
                    elif (int(var) is 1):
                        new_img = img2[:, 0:X]
                        cv2.imwrite(path_new, new_img)
cv2.destroyAllWindows()

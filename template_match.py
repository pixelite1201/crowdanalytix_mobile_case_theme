import cv2
import numpy as np
import os
import hashlib
from matplotlib import pyplot as plt
template = cv2.imread('/home/priyanka/Template/template.jpg',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
meth = 'cv2.TM_CCOEFF_NORMED'
TRAIN_DIR='/home/priyanka/Test/'
count=0
for theme in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, theme)
        for j in os.listdir(path):
            path_new = os.path.join(path,j)
            print(path_new)
            img = cv2.imread(path_new,0)
            img2= cv2.imread(path_new)
            method = eval(meth)
            # Apply template Matching
            try:
                res = cv2.matchTemplate(img,template,method)
                ok = True
            except cv2.error as e:
                ok = False
            if(ok):
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                h_t=top_left[1]+h
                w_t=top_left[0]+w
                crop_img = img[top_left[1]:h_t,top_left[0]:w_t]
                if(np.allclose(template,crop_img,atol=50)):
                    count=count+1
                    cv2.rectangle(img,top_left, bottom_right, 255, 2)
                    #plt.subplot(121),plt.imshow(crop_img,cmap = 'gray')
                    #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                    #plt.subplot(122),plt.imshow(img,cmap = 'gray')
                    #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                    #plt.suptitle(meth)
                    #plt.show()
                    #var=raw_input("1/0")
                    #if(int(var)):
                    new_img=img2[:,0:top_left[0]]
                        #new_img=img2[:,bottom_right[0]:]
                    cv2.imwrite(path_new,new_img)
print(count)

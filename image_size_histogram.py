# Plot histogram of image sizes
import os
import cv2
import matplotlib.pyplot as plt
X=[]
Y=[]
TRAIN_DIR='./Train'
for theme in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, theme)
        for j in os.listdir(path):
            path_new = os.path.join(path,j)
            img = cv2.imread(path_new, 0)
            x,y=img.shape
            X.append(x)
            Y.append(y)

plt.figure(figsize=(20,10))
plt.hist2d(X,Y, bins=40)
plt.show()

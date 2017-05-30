

import os
import hashlib
import cv2
from matplotlib import pyplot as plt
labels={}
hashes={}
TRAIN_DIR='/home/priyanka/Train_orig'
FILE='/home/priyanka/output_inception_13493.csv'
TEST_DIR='/home/priyanka/Test_orig'
for theme in os.listdir(TRAIN_DIR):
        path = os.path.join(TRAIN_DIR, theme)
        for j in os.listdir(path):
            path_new = os.path.join(path,j)
            labels[path_new]=path_new.split('/')[-2]
            h = hashlib.md5(open(path_new, 'rb').read()).hexdigest()
            if h in hashes:
                hashes[h].append(path_new)
            else:
                hashes[h] = [path_new]

with open(FILE) as f:
    content=f.readlines()

test_img=[]
labels_test=[]
for i in content:
    test_img.append(i.split(',')[0])
    labels_test.append(i.split(',')[1])

count=0
count2=0
count3=0
for img,labels_test in zip(test_img,labels_test):
    path=os.path.join(TEST_DIR,img)
    h = hashlib.md5(open(path, 'rb').read()).hexdigest()
    if h in hashes:
        count2=count2+1
        #print("Test img label for ",img, "is ",labels_test)
        if len(hashes[h]) is 1 and labels_test != (labels[hashes[h][0]]+'\n'):
            print("#### ", img,labels_test,labels[hashes[h][0]]+'\n')
            count = count + 1
        #if len(hashes[h]) is 1 and labels_test == (labels[hashes[h][0]]+'\n'):
            #count3 = count3 + 1
        #for x in hashes[h]:
            #print("Train img label for ", x, "is", labels[x])

print count
print count2
print count3
            #plt.subplot(121), plt.imshow(cv2.imread(path), cmap='gray')
            #plt.title('Test_img %s %s'%(labels_test,img)), plt.xticks([]), plt.yticks([])
            #plt.subplot(122), plt.imshow(cv2.imread(x), cmap='gray')
            #plt.title('Train img %s %s'%(labels[x],x.split('/')[-1])), plt.xticks([]), plt.yticks([])
            #plt.show()


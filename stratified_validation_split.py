import os
import hashlib
import cv2
import numpy as np
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
labels={}
hashes={}
TRAIN_DIR='/home/priyanka/Train'
VALIDATION = '/home/priyanka/Validation'
if not os.path.exists(VALIDATION):
    os.makedirs(VALIDATION)
for theme in os.listdir(TRAIN_DIR):
        path_train = os.path.join(TRAIN_DIR, theme)
        path_val = os.path.join(VALIDATION,theme)
        if not os.path.exists(path_val):
            os.makedirs(path_val)
        for j in os.listdir(path_train):
            path_new = os.path.join(path_train,j)
            labels[path_new]=path_new.split('/')[-2]
            h = hashlib.md5(open(path_new, 'rb').read()).hexdigest()
            if h in hashes:
                hashes[h].append(path_new)
            else:
                hashes[h] = [path_new]
X=[]
Y=[]
for k,v in hashes.items():
    if len(v) == 1:
        X.append(hashes[k])
        Y.append(labels[hashes[k][0]])

X=np.array(X)
Y=np.array(Y)
sss = StratifiedShuffleSplit(test_size=0.2,random_state=0)
test_index=[]
for tr_i,te_i in sss.split(X,Y):
    X_test=X[te_i]
    Y_test=Y[te_i]

for x in X_test:
    theme=x[0].split('/')[-2]
    dest_dir = os.path.join(VALIDATION,theme)
    shutil.move(x[0],dest_dir)




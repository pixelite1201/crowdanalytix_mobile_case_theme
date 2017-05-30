import os
import hashlib
import cv2
labels={}
hashes={}
TRAIN_DIR='/home/priyanka/Val_Download'
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
print('Identical files with different labels:')
for k,v in hashes.items():
    if len(v) > 1:
        c = set([labels[x] for x in v])
        if len(c) == 1:
            image_rem=hashes[k][:-1]
            for x in image_rem:
                print x
                os.remove(x)


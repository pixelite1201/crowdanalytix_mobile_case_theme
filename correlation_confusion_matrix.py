import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')
DIR='/home/priyanka/crowdanalytix/all_sumbissions/all_diff/'
list_files=[]
for file in os.listdir(DIR):
    list_files.append(os.path.join(DIR,file))

X=list(itertools.combinations(list_files,2))
for FILE1,FILE2 in X:
    Y_file1=[]
    Y_file2=[]
    with open(FILE1,'r') as myfile:
        next(myfile)
        for line in myfile:
            Y=line.split(',')[-1]
            Y_file1.append(Y.replace('\n',''))

    with open(FILE2,'r') as myfile:
        next(myfile)
        for line in myfile:
            Y=line.split(',')[-1]
            Y_file2.append(Y.replace('\n',''))
    Y_file1=np.array(Y_file1)
    Y_file2=np.array(Y_file2)
    output = confusion_matrix(Y_file1,Y_file2)
    with open('all_diff_conf_matrix','a') as outfile:
        outfile.write(FILE1.split('/')[-1]+" "+FILE2.split('/')[-1]+"    "+
                      str(1200-np.trace(output))+'\n')
#df_cm = pd.DataFrame(output)
#plt.figure(figsize = (10,7))
#sn.heatmap(df_cm, annot=True)
#plt.show()
#savefig('inception_augment_confusion')

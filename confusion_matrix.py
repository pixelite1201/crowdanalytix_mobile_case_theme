## Create Confusion matrix using file cotaining true value and predicted value for a label
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
DIR='./output_inception_my_loss' #Directory containing file having true and predicted value
TRAIN='./Train'
Y_true=[]
Y_pred=[]
label=[]
for theme in os.listdir(TRAIN):
    label.append(theme)
label.sort()
for file in os.listdir(DIR):
    path=os.path.join(DIR,file)
    with open(path,'r') as myfile:
        for line in myfile:
                    Y=line.split(' ')[4]
                    Y_true.append(Y.replace('\n',''))
                    Y_pred.append(line.split(' ')[2])

Y_true=np.array(Y_true)
Y_pred=np.array(Y_pred)
output = confusion_matrix(Y_true,Y_pred)
print output

df_cm = pd.DataFrame(output, index = [i for i in label],
                  columns = [j for j in label])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

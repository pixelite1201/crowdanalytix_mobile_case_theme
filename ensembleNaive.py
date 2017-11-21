#Taking the maximum vote from different trained classifier result
import os
import pandas as pd
import numpy as np

from scipy.stats import mode

list_files = ['./all_sumbissions/70-75/new_ensemblesubmission.csv',
              './all_sumbissions/60-65/sort_out_inception_res_small_528.csv',
                './all_sumbissions/60-65/sort_output_crowd_latest_3013.csv',
              './all_sumbissions/60-65/sort_output_crowd_latest_4192.csv',
              './all_sumbissions/60-65/sort_out_vgg_my_loss_2358.csv',
            './all_sumbissions/60-65/sort_submission34.csv']



outputs = []

for file in list_files:
    pathFile = file
    df = pd.read_csv(pathFile)
    print file
    imagesNames = df['id'].tolist()
    outputs.append(df['Mobile_Theme'].tolist())

f = open('ensemble_verify_'+'submission.csv', 'w')

f.write('id,Mobile_Theme\n')
for preds in zip(imagesNames, zip(*outputs)):
    print preds, mode(list(preds[1]))[0][0]
    f.write(preds[0] + ',' + mode(list(preds[1]))[0][0])
    f.write('\n')

f.close()
print sorted

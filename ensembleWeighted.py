import os
import pandas as pd
import numpy as np

from scipy.stats import mode
import operator


def getWeightDic(weightFile):
    with open(weightFile) as f:
        return {line.split(",")[0]:float(line.split(",")[1]) for line in f.readlines()}


def weightedEnsemble(folderSubs, weightFile):
    weightDic = getWeightDic(weightFile)

    weightedOuputs = {}

    for file in os.listdir(folderSubs):
        pathFile = os.path.join(folderSubs, file)
        df = pd.read_csv(pathFile)
        imagesNames = df['id'].tolist()
        classes = df['Mobile_Theme'].tolist()

        for imageName, cl in zip(imagesNames, classes):
            if imageName not in weightedOuputs:
                weightedOuputs[imageName] = {cl: weightDic[file]}
            else:
                if cl in weightedOuputs[imageName]:
                    oldWt = weightedOuputs[imageName][cl]
                    newWt = oldWt + weightDic[file]
                    weightedOuputs[imageName][cl] = newWt
                else:
                    weightedOuputs[imageName][cl] = weightDic[file]

    best_class = {k : max(v.iteritems(), key=operator.itemgetter(1))[0] for k, v in weightedOuputs.items()}

    f = open("/home/priyanka/crowdanalytix/all_sumbissions/67-70-weighted_ensemble.csv", 'w')
    f.write('id,Mobile_Theme\n')
    for k, v in best_class.items():
        f.write(k + ',' + v)
        f.write('\n')
    f.close()

if __name__ == "__main__":
    folderSubs = "/home/priyanka/crowdanalytix/all_sumbissions/65-70"
    weightFile = "/home/priyanka/crowdanalytix/all_sumbissions/65-70-accuracy-map.csv"
    weightedEnsemble(folderSubs, weightFile)
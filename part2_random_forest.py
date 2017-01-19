import string
import operator
import os

import numpy   as np
from numpy import array
import re
from pyspark import SparkContext
from pyspark.sql import SQLContext

import math
from math import floor

import pandas as pd
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

from time import time
def drop_header(index, itr):
    if index == 0:
        return iter(list(itr)[1:])
    else:
        return itr

def parseData(line):
    list = line.split(",")
	dates = re.sub(r'\W+', ' ', line[0]).split()
	year, month, date, hour, min, second = dates[0], dates[1], dates[2], dates[3], dates[4], dates[5]
	
    if '"' in list[2]:
        # some of the descripts contain '"'
        for i in xrange(3,len(list)):
            if '"' in list[i]:
                lastquote = i; break;
        list[2:lastquote+1] = [','.join(list[2:lastquote+1])]
    if '"' in list[5]:
        # some of the descripts contain '"'
        for i in xrange(6,len(list)):
            if '"' in list[i]:
                lastquote = i; break;
        list[5:lastquote+1] = [','.join(list[5:lastquote+1])]
    list[1:7] = [string.capwords(str.replace('"','')).replace("Of ", "of ") for str in list[1:7]]
    return list

def transformData(line):
   return [line[1], re.sub(r'\W+', ' ', line[0]).split()[3], line[3], line[4], line[6]]

def get_value(element):
    return element[1]

def save_file(x, y, filename):
    f = open('results/Part1-'+filename + '.txt', 'w')
    for i in range(len(x)):
        f.write(x[i] + ": " + str(y[i]) + "\n")
    f.close() 

def create_labeled_point(line):
    clean_line = line

    # convert to numeric categorical variable
    try: 
        clean_line[1] = feat1.index(clean_line[1])
    except:
        clean_line[1] = len(feat1)

    try:
        clean_line[2] = feat2.index(clean_line[2])
    except:
        clean_line[2] = len(feat2)

    try:
        clean_line[3] = feat3.index(clean_line[3])
    except:
        clean_line[3] = len(feat3)

    try:
        clean_line[4] = feat4.index(clean_line[4])
    except:
        clean_line[4] = len(feat4)

    label = label_set.index(clean_line[0])

    return LabeledPoint(label, array([float(x) for x in clean_line[1:]]))

if __name__ == '__main__':
   sc = SparkContext(appName='Crime Classification Part 2-3 Random Forest')
   sqlContext = SQLContext(sc)
   rdd0 = sc.textFile('train.csv').map(parseData).filter(lambda line: len(line)>1)

   header = rdd0.first()
   # print header
   rdd = rdd0.mapPartitionsWithIndex(drop_header)
   
   if not os.path.exists('results'):
      os.makedirs('results')
   print "====================Data Reading Finished===================="

   # converting RDD to dataframe
   raw_training = rdd.map(transformData)

   feat1 = raw_training.map(lambda x: x[1]).distinct().collect()
   feat2 = raw_training.map(lambda x: x[2]).distinct().collect()
   feat3 = raw_training.map(lambda x: x[3]).distinct().collect()
   feat4 = raw_training.map(lambda x: x[4]).distinct().collect()
   label_set = sorted(raw_training.map(lambda x: x[0]).distinct().collect())
   trainingData = raw_training.map(create_labeled_point)

   dictCat = {}
   dictCatInv = {}
   val = 0
   for key in label_set:
      dictCat[key] = val
      dictCatInv[val] = key
      val += 1
   # for key in label_set:
      # print key, dictCat[key]
   # raw_training = raw_training.map(lambda x: [x[0], x[1], x[2], x[3], x[4]])
   raw_training_df = raw_training.toDF(("category", "hour", "day", "PdDistrict", "address"))
   raw_training_df.show()
   label_col = "category"

   # Train the model
   t0 = time()
   rf_classifier = RandomForest.trainClassifier(trainingData,numClasses=len(label_set),categoricalFeaturesInfo={0: len(feat1), 1: len(feat2), 2: len(feat3),3: len(feat4)},numTrees=50, featureSubsetStrategy="auto",impurity='entropy', maxDepth=15, maxBins=max([len(feat1),len(feat2),len(feat3),len(feat4)]))
   tt = time() - t0
   print "Classifier trained in {} seconds".format(round(tt,3))

   # Evaluate the model
   testData = trainingData
   predictions = rf_classifier.predict(testData.map(lambda p: p.features))
   labelsAndPredictions = testData.map(lambda p: p.label).zip(predictions)

   t0 = time()
   testAccuracy = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(testData.count())
   tt = time() - t0
   print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(testAccuracy,4))

   file = open('results/Part2-random forest.txt','w')
   file.write("True Label,  Predicted Label\n")
   # for (true_label, predict_label) in labelsAndPredictions.collect():
      # print (dictCatInv[int(true_label)], dictCatInv[int(predict_label)])
   testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
   print('Test Rate = ' + str(testErr))
   file.write('Test Rate = ' + str(testErr)+'\n')
   for (true_label, predict_label) in labelsAndPredictions.collect():
      file.write(str(true_label==predict_label)+': '+dictCatInv[int(true_label)]+', ' +dictCatInv[int(predict_label)]+'\n')
   # print('Learned classification forest model:')
   # print(rf_classifier.toDebugString())

file.close()

sc.stop()
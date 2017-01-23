import string
import operator
import os
import re
import math
from math import floor
from time import time

import numpy as np
from numpy import array
import pandas as pd
import matplotlib
import matplotlib as mpl
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
# from spark.mllib.tree import configuration.Strategy
# from sklearn.ensemble import RandomForestClassifier
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def drop_header(index, itr):
    if index == 0:
        return iter(list(itr)[1:])
    else:
        return itr

def parseData(line):
    list = line.split(",")
    if '"' in list[2]:
        # some of the descripts contain '"'
        for i in xrange(3, len(list)):
            if '"' in list[i]:
                lastquote = i; break;
        list[2:lastquote+1] = [','.join(list[2:lastquote+1])]
    if '"' in list[5]:
        # some of the descripts contain '"'
        for i in xrange(6, len(list)):
            if '"' in list[i]:
                lastquote = i; break;
        list[5:lastquote+1] = [','.join(list[5:lastquote+1])]
    list[1:7] = [string.capwords(str.replace('"','')).replace("Of ", "of ") for str in list[1:7]]
    return list

# (class, (hour, day, month, year, police district)) independent
def transformData(line): # r'\W+', ' ': replace non-alphanumeric with space
   #return [line[1], re.sub(r'\W+', ' ', line[0]).split()[3], line[3], line[4], line[6]] # old features
   return [line[1], re.sub(r'\W+', ' ', line[0]).split()[3], line[3], re.sub(r'\W+', ' ', line[0]).split()[0:1], line[4]]

def get_value(element):
    return element[1]

def save_file(x, y, filename):
    f = open("results/Part1-" + filename + ".txt", 'w')
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
   sc = SparkContext(appName='Crime Classification Part 2-2 DT')
   sqlContext = SQLContext(sc)
   rdd0 = sc.textFile('train_35000.csv').map(parseData).filter(lambda line: len(line)>1)

   header = rdd0.first()
   # print(header)
   rdd = rdd0.mapPartitionsWithIndex(drop_header)
   
   if not os.path.exists('results'):
      os.makedirs('results')
   print("====================Data Reading Finished====================")

   # converting RDD to dataframe
   raw_training = rdd.map(transformData)

   feat1 = raw_training.map(lambda x: x[1]).distinct().collect()
   feat2 = raw_training.map(lambda x: x[2]).distinct().collect()
   feat3 = raw_training.map(lambda x: x[3]).distinct().collect()
   feat4 = raw_training.map(lambda x: x[4]).distinct().collect()
   feat5 = raw_training.map(lambda x: x[5]).distinct().collect()
   label_set = sorted(raw_training.map(lambda x: x[0]).distinct().collect())
   parsedData = raw_training.map(create_labeled_point)
   trainingData, testData = parsedData.randomSplit([0.6, 0.4], seed=11L)
   
   # category = sorted(raw_training.map(lambda x: x[0]).distinct().collect())
   dictCat = {}
   dictCatInv = {}
   val = 0
   for key in label_set:
      dictCat[key] = val
      dictCatInv[val] = key
      val += 1
   raw_training_df = raw_training.toDF(("category", "hour", "day", "PdDistrict", "address"))
   raw_training_df.show()
   label_col = "category"

   # Train the model
   t0 = time()
   dt_classifier = DecisionTree.trainClassifier(trainingData,numClasses=len(label_set),categoricalFeaturesInfo={0: len(feat1), 1: len(feat2), 2: len(feat3),3: len(feat4), 4:len(feat5)},impurity='gini', maxDepth=30, maxBins=max([len(feat1),len(feat2),len(feat3),len(feat4)]))
   tt = time() - t0
   print "Classifier trained in {} seconds".format(round(tt, 3))

   # Evaluate the model
   # testData = trainingData
   predictions = dt_classifier.predict(testData.map(lambda p: p.features))
   labelsAndPredictions = testData.map(lambda p: p.label).zip(predictions)

   t0 = time()
   testAccuracy = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(testData.count())
   tt = time() - t0
   print("Prediction made in {} seconds. Test accuracy is {}".format(round(tt, 3), round(testAccuracy, 4)))

   file = open('results/Part2-decision tree.txt','w')
   file.write("True Label,  Predicted Label\n")
   # for (true_label, predict_label) in labelsAndPredictions.collect():
      # print(dictCatInv[int(true_label)], dictCatInv[int(predict_label)])
   testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
   print('Test Rate = ' + str(testErr))
   file.write('Test Rate = ' + str(testErr) + '\n')
   for (true_label, predict_label) in labelsAndPredictions.collect():
      file.write(str(true_label==predict_label) + ': ' + dictCatInv[int(true_label)] + ', ' + dictCatInv[int(predict_label)] + '\n')
   # print('Learned classification forest model:')
   # print(dt_classifier.toDebugString())
   # Save and load model
   # model.save(sc, "myModelPath")
   # sameModel = DecisionTreeModel.load(sc, "myModelPath")

file.close()

sc.stop()

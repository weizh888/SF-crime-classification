import os
from time import time
import itertools

from numpy import array

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

if __name__ == '__main__':
   sc = SparkContext(appName='Crime Classification Part 2-1 LR')
   sqlContext = SQLContext(sc)
   raw_training = sc.textFile('Part2-numerical dataset.csv').map(lambda line: line.split(",")).filter(lambda line: len(line)>1)
   
   if not os.path.exists('results'):
      os.makedirs('results')
   print("====================Data Reading Finished====================")

   feat1 = raw_training.map(lambda x: x[0]).distinct().collect()
   feat2 = raw_training.map(lambda x: x[1]).distinct().collect()
   feat3 = raw_training.map(lambda x: x[2]).distinct().collect()
   feat4 = raw_training.map(lambda x: x[3]).distinct().collect()
   label_set = sorted(raw_training.map(lambda x: x[4]).distinct().collect())
   parsedData = raw_training.map(lambda x: LabeledPoint(x[4], array([float(x) for x in x[0:4]])))
   trainData, testData = parsedData.randomSplit([0.6, 0.4], seed=11L)
   
   dictCat = {}
   dictCatInv = {}
   val = 0
   for key in label_set:
      dictCat[key] = val
      dictCatInv[val] = key
      val += 1
   raw_training_df = raw_training.toDF(("hour", "day", "PdDistrict", "address", "category"))
   raw_training_df.show()
   label_col = "category"    
 
   # Train the model 
   t0 = time()     
   lr_classifier = LogisticRegressionWithLBFGS.train(trainData, regParam=1, intercept=True, numClasses=len(label_set))
   tt = time() - t0
   print("Classifier trained in {} seconds".format(round(tt, 3)))
 
   predictions = lr_classifier.predict(testData.map(lambda x: x.features))
   labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions) 

   t0 = time()   
   testErr = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1]).count() / float(testData.count())
   tp = time() - t0
   print("Prediction made in {} seconds. Test error is {}".format(round(tp, 3), round(testErr, 4)))
   
   file = open('results/Part2-logistic reg.txt','w')
   file.write("True Label,  Predicted Label\n")
   for (true_label, predict_label) in labelsAndPredictions.collect():
      print(dictCatInv[int(true_label)], dictCatInv[int(predict_label)])
      file.write(dictCatInv[int(true_label)] + ': ' + dictCatInv[int(predict_label)] + '\n')

   print('Error Rate = ' + str(testErr))
   file.write("Classifier trained in {} seconds".format(round(tt, 3)) + '\n')
   file.write("Prediction made in {} seconds".format(round(tp, 3)) + '\n')
   file.write("Test error is {}".format(round(testErr, 4)))

file.close()
sc.stop()

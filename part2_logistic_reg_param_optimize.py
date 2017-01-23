import os

from numpy import array
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
import itertools

if __name__ == '__main__':
   sc = SparkContext(appName='Crime Classification Part 2-1 LR-param')
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
   trainData, testData = parsedData.randomSplit([0.6, 0.4] , seed=11L)
   
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
   
   regParamSet = [10**e for e in range(-5, 6)]
   interceptSet = [True, False]   
   bestModel = None
   bestTestErr = float("inf")
   bestRegParam = -1.0
   bestIntercept = False
   for r, intcp in itertools.product(regParamSet, interceptSet):
       lr_classifier = LogisticRegressionWithLBFGS.train(trainData, regParam=r, intercept=intcp, numClasses=len(label_set))
       predictions = lr_classifier.predict(testData.map(lambda x: x.features))
       labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
       testErr = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1]).count() / float(testData.count())
       if (testErr < bestTestErr):
           bestModel = lr_classifier
           bestTestErr = testErr
           bestRegParam = r
           bestIntercept = intcp
   
   file = open('results/Part2-logistic reg param.txt','w')
   print('Best Error Rate = ' + str(bestTestErr))
   print('Best RegParam = ' + str(bestRegParam))
   print('Best Intercept = ' + str(bestIntercept))
   file.write('Best Error Rate = ' + str(bestTestErr) + '\n')
   file.write('Best RegParam = ' + str(bestRegParam) + '\n')
   file.write('Best Intercept = ' + str(bestIntercept) + '\n')

file.close()
sc.stop()

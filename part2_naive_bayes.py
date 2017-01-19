# from pyspark import SparkContext
import csv
import random
import math
from math import log
 
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
 
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def parseLine(line):
  list = line.split(',')
  return [float(x) for x in list]

if __name__ == '__main__':
   # sc = SparkContext(appName='Crime Classification Part 2-3')
   # dataset = sc.textFile('results/Part2-numerical dataset_s.csv').map(parseLine)
   # # # for i in dataset.take(20):
    # # # print i
   # trainingData, testData = dataset.randomSplit([0.7, 0.3], seed = 0)
   # trainingSet = trainingData.collect()
   # testSet = testData.collect
   # # # for i in trainingData.collect():
     # # # print i
   # # print trainingData.count(), testData.count()
   # # classes = trainingData.map(lambda x: x[4]).distinct().collect()

   filename = 'results/Part2-numerical dataset.csv'
   splitRatio = 0.7
   dataset = loadCsv(filename)
   for i in dataset[1:20]:
      print i
   
   trainingSet, testSet = splitDataset(dataset, splitRatio)
   print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
   # prepare model

   separated = separateByClass(trainingSet)
   # print separated
   # print '\n'
   # separated2 = trainingData.map(lambda x: x[4]).distinct().collect()
   # print separated2
   # summaries = {}
   # for classValue, instances in separated.iteritems():
      # summaries[classValue] = summarize(instances)
   # for i in summaries:
      # print i, summaries[i]

   summaries = summarizeByClass(trainingSet)
   # for i in summaries:
        # print i, summaries[i]
   # # test model
   predictions = []
   logloss = 0
   for i in range(len(testSet)):
      result = predict(summaries, testSet[i])

      probabilities = calculateClassProbabilities(summaries, testSet[i])
      trueClass_j = testSet[i][-1]
      if trueClass_j in probabilities:
         p_ij = probabilities[trueClass_j]
      else:
         p_ij = 0
      if p_ij < 10^(-15):
         p_ij = 10^(-15)
      if p_ij > 1-10^(-15):
         p_ij = 1-10^(-15)
      print p_ij
      logloss += 1*math.log(p_ij)
   print logloss
   print -logloss*1.0/len(testSet)
      # print probabilities
      # bestLabel, bestProb = None, -1
      # for classValue, probability in probabilities.iteritems():
         # if bestLabel is None or probability > bestProb:
            # bestProb = probability
            # bestLabel = classValue

'''
      predictions.append(result)
   # predictions = getPredictions(summaries, testSet)
   # accuracy = getAccuracy(testSet, predictions)
   # print('Accuracy: {0}%').format(accuracy)
'''
# sc.strop()

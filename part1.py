import string
import operator
import os

# import pandas  as pd
import numpy   as np
# import seaborn as sns
import re
from pyspark import SparkContext

import matplotlib
import matplotlib as mpl
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import math
from math import floor

from pyspark.mllib.fpm import FPGrowth

# # Plotting Options
# sns.set_style("whitegrid")
# sns.despine()

def drop_header(index, itr):
    if index == 0:
        return iter(list(itr)[1:])
    else:
        return itr

def parseData(line):
    list = line.split(",")
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

def barplot(x, y, fontsize, filename):
    # plt.rc('font', weight='bold')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos = np.arange(len(x))+0.5
    colorList = [cm.gist_rainbow(1.0*i/len(x)+0.1) for i in range(0,len(x))]
    ax.bar(pos, y, align='center', color=colorList, width=0.75)
    ax.set_xticks(pos)
    ax.set_xticklabels(x)
    ax.set_xlim(0, len(x))
    # Set the tick labels font
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
    fig.autofmt_xdate()
    plt.tight_layout()
    ax.set_title('Crime frequency of different '+filename)
    plt.savefig('results/crime-'+filename+'.png', dpi=300)

def parseTime(line):
    time = line[0]
    list = re.sub(r'\W+', ' ', time).split()
    hour = float(list[3])
    # intvl is a global variable
    return (math.floor(hour*1.0/intvl), 1) # count every 2 hours

def parseAddress(line):
    address = line[6]
    if '/' in address:
        crossSts = address.split('/')
        return [[crossSts[0].strip(), 1],[crossSts[1].strip(), 1]] # .strip() to remove leading and trailing whitespace
    else:
        return [[address, 1]]

def get_value(element):
    return element[1]

def save_file(x, y, filename):
    f = open('results/Part1-'+filename + '.txt', 'w')
    for i in range(len(x)):
        f.write(x[i] + ": " + str(y[i]) + "\n")
    f.close() 

if __name__ == '__main__':
    sc = SparkContext(appName='Crime Classification Part 1')
    rdd0 = sc.textFile('train.csv').map(parseData).filter(lambda line: len(line)>1)
    header = rdd0.first()
    print header
    rdd = rdd0.mapPartitionsWithIndex(drop_header)
    # for i in rdd.take(50):
        # print i
    
    if not os.path.exists('results'):
        os.makedirs('results')
    print "====================Data Reading Finished===================="

    category_sorted = rdd.map(lambda a: (a[1],1)).reduceByKey(lambda a, b: a+b).sortBy(get_value, ascending=False)
    x = [key for (key, val) in category_sorted.collect()]
    y = [val for (key, val) in category_sorted.collect()]
    barplot(x, y, 7, 'categories')
    save_file(x, y, 'categories')
    print "====================Frequency of Categories Plotted===================="

    dayOfWeek = rdd.map(lambda a: (a[3],1)).reduceByKey(lambda a, b: a+b).collectAsMap()
    # print len(dayOfWeek)
    # for i in dayOfWeek:
        # print i
    weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    x = [i for i in weekDays]
    y = [dayOfWeek[key] for key in weekDays]
    barplot(x, y, 12, 'days')
    save_file(x, y, 'days')
    print "====================Frequency on Days Plotted===================="

    PdDistrict_sorted = rdd.map(lambda a: (a[4],1)).reduceByKey(lambda a, b: a+b).sortBy(get_value, ascending=False)
    x = [key for (key, val) in PdDistrict_sorted.collect()]
    y = [val for (key, val) in PdDistrict_sorted.collect()]
    barplot(x, y, 12, 'PdDistricts')
    save_file(x, y, 'PdDistricts')
    print "====================Frequency in PdDistrict Plotted===================="

    resolution_sorted = rdd.map(lambda a: (a[5],1)).reduceByKey(lambda a, b: a+b).sortBy(get_value, ascending=False)
    x = [key for (key, val) in resolution_sorted.collect()]
    y = [val for (key, val) in resolution_sorted.collect()]
    barplot(x, y, 9, 'resolutions')
    save_file(x, y, 'resolutions')
    print "====================Frequency of Resolutions Plotted===================="

    # address_sorted = rdd.flatMap(parseAddress).reduceByKey(lambda a, b: a+b).sortBy(get_value, ascending=False)
    address_sorted = rdd.map(lambda a: ((a[6],float(a[7]),float(a[8])),1)).reduceByKey(lambda a, b: a+b).sortBy(get_value, ascending=False)
    # bar plot of Top 25
    x = [key[0] for (key, val) in address_sorted.take(25)]
    y = [val    for (key, val) in address_sorted.take(25)]
    barplot(x, y, 8, 'addresses')
    save_file(x, y, 'addresses')
    print "====================Frequency at Addresses Plotted===================="

    for intvl in range(1,4):
        time_sorted = rdd.map(parseTime).reduceByKey(lambda a, b: a+b).sortByKey(ascending=True)
        x = [str(int(intvl*key))+'~'+str(int(intvl*(key+1))) for (key, val) in time_sorted.collect()]
        y = [val for (key, val) in time_sorted.collect()]
        barplot(x, y, 10, 'hours (intvl = '+str(intvl)+')')
        save_file(x, y, 'hours (intvl = '+str(intvl)+')')
    print "====================Frequency at Hours Plotted===================="

    x = [key[1] for (key, val) in address_sorted.collect()]
    y = [key[2] for (key, val) in address_sorted.collect()]
    z = [val    for (key, val) in address_sorted.collect()]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colorList = [cm.gist_rainbow(1.0*i/len(x)+0.1) for i in range(0,len(x))]
    ax.scatter(x, y, z, marker='o', color=colorList, s=1)
    ax.set_xlim(-122.6, -122.4)
    ax.set_ylim(38, 42)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Count')
    plt.savefig('results/crime-addresses(3D)_z.png', dpi=300)

    # find Frequent Itemset
    transactions = rdd.map(lambda e: [re.sub(r'\W+',' ',e[0]).split()[3], e[1], e[3], e[4], e[5], e[6]])
    # for i in transactions.take(10):
        # print i
    model = FPGrowth.train(transactions, minSupport=0.015,numPartitions=10)
    result = model.freqItemsets().collect()
    result_filtered = model.freqItemsets().filter(lambda a: len(a[0])>2).collect()
    print('---------------------------')
    c = 0
    for fi in result_filtered:
        c = c + 1
        print(fi)
    print(c)

    # create a dictionary to store the frequency for itemsets
    freDict = {}
    file1 = open('results/Part1-freq itemsets_conf.txt','w')
    file2 = open('results/Part1-freq itemsets_intr.txt','w')
    totalBaskets = transactions.count()
    for fi in result:
       fi[0].sort()
       key = ', '.join(fi[0])
       freDict[key]=fi[1]

    for fi in result_filtered:
       fi[0].sort()
       myStr = ''
       myStr = '{'+', '.join(fi[0])+'}: '+str(fi[1])
       file1.write(myStr+'\n')
    file1.write('\n')

    for fi in result_filtered:
       fi[0].sort()
       if len(fi[0])>1:
          for i in xrange(len(fi[0])):
             j = fi[0][i]
             I_key = ""
             for k in xrange(len(fi[0])):
                if k != i:
                   if I_key == "":
                      I_key = fi[0][k]
                   else:
                      I_key = I_key+', '+fi[0][k]
             #print (I_key)
             confidence = fi[1]*1.0/freDict[I_key]
             interest = confidence - freDict[j]*1.0/totalBaskets
             myStr1 = 'conf({'+I_key+'}=>{'+j+'})='+str(round(confidence,4))
             myStr2 = 'interest({'+I_key+'}=>{'+j+'})='+str(round(interest,4))
             print (myStr1)
             print (myStr2)
             file1.write(myStr1+'\n')
             file2.write(myStr2+'\n')
    file1.close()
    file2.close()

sc.stop()

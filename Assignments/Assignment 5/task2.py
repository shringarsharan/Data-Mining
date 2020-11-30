from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from datetime import datetime
import json, random, time, sys, csv, os
import binascii
from statistics import mean, median
from math import log2
# %%
port_num = int(sys.argv[1])
output_path = sys.argv[2]
# %%
# input_path = 'data/business.json'
# output_path = 'data/task2.res.csv'
# %%
sc = SparkContext('local[*]', 'task2')
sc.setLogLevel("OFF")
ssc = StreamingContext(sc, 5)
ssc.checkpoint(os.getcwd()+'\logs')
# %%
def outputStream(iter):
    header = ['Time', 'Ground Truth', 'Estimation', '%diff']
    with open(output_path, 'a+') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        for time,count in iter:
            writer.writerow({'Time': time, 'Ground Truth': count[0], 'Estimation': count[1], '%diff':abs((count[0]-count[1])/count[0])})

def hashFuncs(num_grps, num_hash):
    random.seed(3)
    return {i: [(random.randint(0, 200), random.randint(0, 200), random.randint(201, 300))
                for j in range(num_hash)] for i in range(num_grps)}

def trailingZeros(x, hashes):
    intx = int(binascii.hexlify(x.encode('utf8')), 16)
    tr0 = {has: [len(bin((a*intx+b)%m)) - len(bin((a*intx+b)%m).rstrip('0')) for a,b,m in hashes[has]] for has in hashes}
    return tr0

def flajoletMartin(x):
    return median(mean(2**i for i in x[has]) for has in x)

def reducefunc(x,y):
    return {grp: [max(x[grp][i],y[grp][i]) for i in range(len(x[grp]))] for grp in x}

def martin2(winrdd):
    window = [int(binascii.hexlify(i.encode('utf8')), 16) for i in winrdd.collect()]
    actual = len(set(window))
    estimate = median(mean([2**max(len(bin((a*intx+b)%m)) - len(bin((a*intx+b)%m).rstrip('0')) for intx in window)
                            for a,b,m in hashes[grp]]) for grp in hashes)
    with open(output_path, 'a+') as f:
        csv.writer(f).writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), actual, estimate])

# %%
with open(output_path, 'a+') as f:
    csv.writer(f).writerow(['Time', 'Ground Truth', 'Estimation'])
hashes = hashFuncs(121, 2)
rdd = ssc.socketTextStream('localhost', port_num).map(json.loads).map(lambda x: x['state'])
# states = rdd.window(30,10).map(lambda x: trailingZeros(x, hashes))\
#     .reduce(lambda x,y: reducefunc(x,y))\
#     .map(lambda x: (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flajoletMartin(x)))
    # .reduceByWindow(lambda x,y: {grp: [max(x[grp][i],y[grp][i]) for i in range(len(x[grp]))] for grp in x},
    #                 lambda x,y: {grp: [min(x[grp][i],y[grp][i]) for i in range(len(x[grp]))] for grp in x},
    #                 windowDuration=30, slideDuration=10)\

# states = rdd.window(30,10).map(lambda x: trailingZeros(x, hashes))\
#     .map(lambda x: (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flajoletMartin(x)))\
#     .reduceByWindow(lambda x,y: {grp: [max(x[grp][i],y[grp][i]) for i in range(len(x[grp]))] for grp in x},
#                     lambda x,y: {grp: [min(x[grp][i],y[grp][i]) for i in range(len(x[grp]))] for grp in x},
#                     windowDuration=30, slideDuration=10)\
#actual = rdd.countByValueAndWindow(30,10).count().map(lambda x: (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), x))

#actual = actual.join(states)

#actual.pprint()
#actual.foreachRDD(lambda x: x.foreachPartition(outputStream))
finrdd = rdd.window(30,10).foreachRDD(martin2)
ssc.start()
ssc.awaitTermination(120)

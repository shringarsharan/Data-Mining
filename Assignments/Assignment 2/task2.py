from pyspark import SparkContext
import time
from itertools import combinations
import itertools
from collections import Counter
import csv
import sys
import time
import os
from operator import add
#%%
s = int(sys.argv[1])
input_path = sys.argv[2]
output_path = sys.argv[3]
#%% Functions
def candidates(k,prevFreq):
    prev = sorted(set(itertools.chain.from_iterable(prevFreq)))
    return combinations(prev, k)

def frequent(k, baskets, candidates, partition_thresh):
    d = Counter()
    for itemset in candidates:
        for basket in baskets:
            if set(itemset).issubset(basket):
                d[itemset] += 1
    freq = [k for k,v in d.items() if v >= partition_thresh]
    return freq

def apriori(baskets,s,total_keys):
    baskets = list(baskets)
    d1 = Counter()
    ct = 0
    for basket in baskets:
        ct += 1
        for item in basket:
            d1[item] += 1
    p = ct/total_keys
    partition_thresh = p*s
    L = sorted([tuple([k]) for k, v in d1.items() if v >= partition_thresh])
    final_freq = []
    final_freq.extend(L)
    k=1
    while len(L) >= 2:
        C = candidates(k+1,L)
        L = sorted(frequent(k+1,baskets,C,partition_thresh))
        final_freq.extend(L)
        k += 1
    return final_freq

def count_frequent(x, first_pass):
    d = Counter()
    for basket in x:
        for itemset in first_pass:
            if set(itemset).issubset(basket):
                d[itemset] += 1
    return list(d.items())
#%%
def task2(rdd, s):
    total_keys = rdd.count()
    first_pass = rdd.mapPartitions(lambda baskets: apriori(baskets,s,total_keys)).distinct()
    first_single = first_pass.filter(lambda x: len(x)==1).sortBy(lambda x: x[0]).collect()
    first_multiple = first_pass.filter(lambda x: len(x)>1).sortBy(lambda x: (len(x),x[0],x[1])).collect()

    first_pass_copy = first_pass.collect()

    second_pass = rdd.mapPartitions(lambda x: count_frequent(x,first_pass_copy)).reduceByKey(add).filter(lambda x: x[1]>=s).map(lambda x: x[0])
    second_single = second_pass.filter(lambda x: len(x)==1).sortBy(lambda x: x[0]).collect()
    second_multiple = second_pass.filter(lambda x: len(x)>1).sortBy(lambda x: (len(x), x[0], x[1])).collect()
    return first_single, first_multiple, second_single, second_multiple
#%%
sc = SparkContext('local[*]', 'task1')
start = time.time()
rdd = sc.textFile(input_path).mapPartitions(lambda x: csv.reader(x))
header = rdd.first()
rdd = rdd.filter(lambda x: x != header)
rdd = rdd.groupByKey().mapValues(lambda x: list(set(x))).map(lambda x: x[1]).persist()

first_single, first_multiple, second_single, second_multiple = task2(rdd,s)

#%%
with open(output_path,"w") as f:
    f.writelines("Candidates:")
    for i in range(1,len(first_multiple[-1])+1):
        if i==1:
            out = "\n"+str(first_single).replace(',)', ')').replace('[','').replace(']','').replace(', ',',')+"\n"
            f.writelines(out)
        if i>1:
            out = sorted(list(filter(lambda x: len(x)==i, first_multiple)))
            out = "\n"+str(out).replace(', (', ',(').replace('[','').replace(']','')+"\n"
            f.writelines(out)
    f.writelines("\nFrequent Itemsets:")
    for i in range(1,len(second_multiple[-1])+1):
        if i==1:
            out = "\n"+str(second_single).replace(',)', ')').replace('[','').replace(']','').replace(', ',',')+"\n"
            f.writelines(out)
        if i>1:
            out = sorted(list(filter(lambda x: len(x)==i, second_multiple)))
            out = "\n"+str(out).replace(', (', ',(').replace('[','').replace(']','')+"\n"
            f.writelines(out)
f.close()
end = time.time()
diff = round(end-start,2)
print("Duration:", diff)
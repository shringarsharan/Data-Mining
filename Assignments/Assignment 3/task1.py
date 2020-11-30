from __future__ import division
from pyspark import SparkContext
from itertools import combinations, islice
import sys
import time
import json
import random
#%%
input_path = sys.argv[1]
output_path = sys.argv[2]
#%%
def newminhash(n,col):
    sig = []
    random.seed(2)
    for i in range(n):
        a,b,m = random.randint(0,10000), random.randint(0,10000), random.randint(20000,30000)
        perm = [(a * i + b) % m for i in col]
        sig.append(min(perm))
    return sig
def banding(x,bands,rows):
    n = iter(range(len(x)))
    k = iter(x)
    elems = [rows]*bands
    slices = [tuple([next(n),tuple(islice(k,i))]) for i in elems]
    return slices
def similarity(x, bizset):
    C0 = set(bizset[x[0]])
    C1 = set(bizset[x[1]])
    jaccard = len(C0.intersection(C1))/len(C0.union(C1))
    return jaccard
#%%
def task1(review):
    users = review.map(lambda x: x['user_id']).distinct().collect()
    d_users = dict(zip(users,range(len(users))))
    review = review.repartition(28).map(lambda x: (x['business_id'], d_users[x['user_id']])).groupByKey().mapValues(lambda x: list(set(x))).persist()

    bizset = dict(review.collect())

    lsh = review.map(lambda x: (x[0],newminhash(50,x[1]))).map(lambda x: (x[0],banding(x[1],50,1))).flatMap(lambda x: [(i,x[0]) for i in x[1]])
    candidates = lsh.groupByKey().mapValues(lambda x: sorted(set(x))).filter(lambda x: len(x[1]) > 1)
    candidates = candidates.flatMap(lambda x: combinations(x[1],2)).distinct()

    sim = candidates.map(lambda x: (x,similarity(x, bizset))).filter(lambda x: x[1] >= 0.055)
    sim = sim.sortBy(lambda x: (x[0][0],x[0][1])).map(lambda x: dict([('b1',x[0][0]),('b2',x[0][1]),('sim',x[1])]))
    return sim.collect()
#%%
sc = SparkContext('local[*]', 'task1')
review = sc.textFile(input_path).map(json.loads)
start = time.time()
output = task1(review)

with open(output_path, 'w') as f:
    for i in output:
        f.write(json.dumps(i) + '\n')
end = time.time()
print(end-start)
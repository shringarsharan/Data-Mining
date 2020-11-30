from __future__ import division
from pyspark import SparkContext
from itertools import combinations, islice, product
from statistics import mean
from collections import Counter, defaultdict
import sys
import time
from math import log2, sqrt
import json
import random

# %%
train_path = sys.argv[1]
model_path = sys.argv[2]
cf_type = sys.argv[3]
# %% User-based CF functions
def userPearsonSimilarity(x, userset):
    u1, u2 = userset[x[0][0]], userset[x[0][1]]
    co_rated_biz = set(x[1][1])
    u1pf, u2pf = dict([(i, u1[i]) for i in co_rated_biz]), dict([(i, u2[i]) for i in co_rated_biz])
    u1avg, u2avg = mean(u1pf.values()), mean(u2pf.values())
    numerator = sum((a - u1avg) * (b - u2avg) for a, b in zip(u1pf.values(), u2pf.values()))
    if numerator <= 0:
        sim = -1
    else:
        denominator = sqrt(sum((rt - u1avg) ** 2 for rt in u1pf)) * sqrt(sum((rt - u2avg) ** 2 for rt in u2pf))
        sim = numerator / denominator
    return {'u1': x[0][0], 'u2': x[0][1], 'sim': sim}


def userMinhashBanding(x, bands, rows):
    sig = []
    N = bands * rows
    random.seed(2)
    for i in range(N):
        a, b, m = random.randint(0, 10000), random.randint(0, 10000), random.randint(20000, 30000)
        perm = [(a * i + b) % m for i in x[1]]
        sig.append(min(perm))
    n, k = iter(range(len(sig))), iter(sig)
    elems = [rows] * bands
    slices = [tuple([next(n), tuple(islice(k, i))]) for i in elems]
    grps = [(i, x[0]) for i in slices]
    return grps


def userJaccardSimilarity(x, userset):
    U0 = set(userset[x[0]])
    U1 = set(userset[x[1]])
    numerator = len(U0.intersection(U1))
    denominator = len(U0.union(U1))
    return (numerator / denominator, U0.intersection(U1))


# %% Item-based CF functions
def itemTransformPairs(x):
    comb_biz = list(combinations(sorted(x), 2))
    ratings = [(x[i[0]], x[i[1]]) for i in comb_biz]
    return list(zip(comb_biz, ratings))


def itemPearsonSimilarity(x):
    b1pf, b2pf = list(zip(*x[1]))
    b1avg, b2avg = mean(b1pf), mean(b2pf)
    numerator = sum((a - b1avg) * (b - b2avg) for a, b in x[1])
    if numerator <= 0:
        similarity = -1
    else:
        denominator = sqrt(sum((rt - b1avg) ** 2 for rt in b1pf)) * sqrt(sum((rt - b2avg) ** 2 for rt in b2pf))
        similarity = numerator / denominator
    return {'b1': x[0][0], 'b2': x[0][1], 'sim': similarity}


#%%
sc = SparkContext('local[*]', 'task3train')
start = time.time()
review = sc.textFile(train_path).map(json.loads)
if cf_type == 'item_based':
    result = review.map(lambda x: (x['user_id'], (x['business_id'], x['stars']))).groupByKey() \
        .mapValues(lambda x: dict(list(x))).flatMap(lambda x: itemTransformPairs(x[1])).groupByKey().mapValues(list) \
        .filter(lambda x: len(x[1]) >= 3) \
        .map(lambda x: itemPearsonSimilarity(x), preservesPartitioning=True) \
        .filter(lambda x: x['sim'] > 0)
else:
    businesses = list(set(review.map(lambda x: x['business_id']).collect()))
    dicBiz = dict(zip(businesses, range(len(businesses))))
    result = review.map(lambda x: (x['user_id'], (dicBiz[x['business_id']], x['stars'])), preservesPartitioning=True) \
        .groupByKey() \
        .mapValues(lambda x: dict(list(set(x)))).filter(lambda x: len(x[1]) >= 3)
    userset = dict(result.collect())
    result = result.flatMap(lambda x: userMinhashBanding(x, 33, 1)).groupByKey().mapValues(lambda x: sorted(set(x))) \
        .filter(lambda x: len(x[1]) > 1).flatMap(lambda x: combinations(x[1], 2)).distinct() \
        .map(lambda x: (x, userJaccardSimilarity(x, userset)), preservesPartitioning=True) \
        .filter(lambda x: (x[1][0] >= 0.01) and (len(x[1][1]) >= 3)) \
        .map(lambda x: userPearsonSimilarity(x, userset), preservesPartitioning=True) \
        .filter(lambda x: x['sim'] > 0)

def writeJSON(x):
    with open(model_path, 'a+') as f:
        for i in iter(x):
            f.write(json.dumps(i) + '\n')

result.foreachPartition(writeJSON)

end = time.time()
print(end - start)

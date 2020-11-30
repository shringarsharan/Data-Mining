from __future__ import division
from pyspark import SparkContext
#from itertools import combinations, islice, product
from statistics import mean
#from collections import Counter, defaultdict
import sys
import time
from math import log2, sqrt
import json
#import random
import os
#%%
train_path = sys.argv[1]
test_path = sys.argv[2]
model_path = sys.argv[3]
output_path = sys.argv[4]
cf_type = sys.argv[5]
#%%
def itemBasedPrediction(x, N, model):
    target, ratings = x[1][0], x[1][1]
    num_biz_user = len(ratings)
    if num_biz_user != 0:  # old user
        pairs = [tuple(sorted([target, i])) for i in ratings]
        weights = [model[i] if i in model.keys() else 0 for i in pairs]
        if sum(weights) == 0:
            predicted = -1
        else:
            if num_biz_user <= N:
                topN = list(zip(pairs, weights, ratings.values()))
            else:
                topN = sorted(zip(pairs, weights, ratings.values()), key=(lambda x: x[1]), reverse=True)[:N]
            predicted = sum(b * c for a, b, c in topN) / sum(b for a, b, c in topN)
    else:
        predicted = -1
    return {'user_id': x[0], 'business_id': target, 'stars': predicted}

def userBasedPrediction(x, N, model, userset):
    target_user, biz_ratings = x[1][0], x[1][1]
    num_user_biz = len(biz_ratings)
    if num_user_biz == 0:  # new business : no ratings
        predicted = -1
    else:    # old business (rated by some users)
        in_pairs = [tuple([target_user, i]) for i in biz_ratings]
        co_rated = dict([(u2, set(userset[u1]).intersection(set(userset[u2]))) for u1, u2 in in_pairs])
        co_rated = dict([(a,b) for a,b in co_rated.items() if b != set()])
        if co_rated == {}:      # No co-rated users
            predicted = -1
        else:
            pairs = [tuple(sorted([target_user, user])) for user in co_rated.keys()]
            weights = [model[i] if i in model.keys() else 0 for i in pairs]
            if sum(abs(wt) for wt in weights) == 0:       # New user: no rating or too few co-rated businesses
                predicted = -1
            else:
                co_rated_avg = [mean(userset[user][biz] for biz in cor_biz) for user, cor_biz in co_rated.items()]
                biz_ratings = [biz_ratings[user] for user in co_rated.keys()]
                if num_user_biz <= N:
                    topN = zip(biz_ratings, co_rated_avg, weights)
                else:
                    topN = sorted(zip(biz_ratings, co_rated_avg, weights), key=(lambda x: x[2]), reverse=True)[:N]
                try:
                    target_user_avg = mean(userset[target_user].values())
                    numerator = sum(((a - b) * c) for a, b, c in topN)
                    denominator = sum(abs(c) for a, b, c in topN)
                    predicted = target_user_avg + numerator/denominator
                except:
                    predicted = -1
    return {'user_id': target_user, 'business_id': x[0], 'stars': predicted}
#%%
sc = SparkContext('local[*]', 'task3predict')
start = time.time()

if cf_type == 'item_based':
    test = sc.textFile(test_path).map(json.loads).map(lambda x: (x['user_id'], x['business_id']))
    model = dict(sc.textFile(model_path).map(json.loads).map(lambda x: ((x['b1'], x['b2']), x['sim'])).collect())
    train = sc.textFile(train_path).map(json.loads).map(lambda x: (x['user_id'], (x['business_id'],x['stars'])))\
        .groupByKey().mapValues(dict)
    result = test.join(train).map(lambda x: itemBasedPrediction(x, 5, model), preservesPartitioning=True)\
        .filter(lambda x: x['stars'] > 0)

elif cf_type == 'user_based':
    test = sc.textFile(test_path).map(json.loads).map(lambda x: (x['business_id'], x['user_id']))
    model = dict(sc.textFile(model_path).map(json.loads).map(lambda x: ((x['u1'], x['u2']), x['sim'])).collect())
    
    #train = sc.textFile(train_path).map(json.loads).map(lambda x: (x['business_id'], (x['user_id'], x['stars']))) \
    #    .groupByKey().mapValues(dict)

    train = sc.textFile(train_path).map(json.loads).map(lambda x: ((x['business_id'],x['user_id']),
                                                        (x['stars'], x['text']))).distinct().mapValues(lambda x: x[0])\
        .aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),lambda a,b: (a[0] + b[0], a[1] + b[1]))\
        .mapValues(lambda v: v[0]/v[1]).map(lambda x: (x[0][0], (x[0][1], x[1])))

    userset = dict(train.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey()\
        .mapValues(lambda x: dict(list(set(x)))).collect())  # .filter(lambda x: len(x[1]) >= 3)

    train = train.groupByKey().mapValues(dict)
    
    result = test.join(train).map(lambda x: userBasedPrediction(x, 50, model, userset), preservesPartitioning=True)\
        .filter(lambda x: x['stars'] > 0)
        
def writeJSON(x):
    with open(output_path, 'a+') as f:
        for i in iter(x):
            f.write(json.dumps(i) + '\n')

result.foreachPartition(writeJSON)     
    
end = time.time()
print(end-start)
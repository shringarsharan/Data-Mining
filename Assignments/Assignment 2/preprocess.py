from pyspark import SparkContext
import time
from itertools import combinations
import itertools
from collections import Counter
import csv
import json
import io
import sys
import time
import os
from operator import add
#%%
sc = SparkContext('local[*]', 'task1')
#%%
review = sc.textFile('data/review.json').map(json.loads).map(lambda x: (x['business_id'], x['user_id']))
biz = sc.textFile('data/business.json').map(json.loads).filter(lambda x: x['stars'] >= 4)
biz = biz.map(lambda x: (x['business_id'], x['state']))

header = sc.parallelize([('user_id','state')])
joined = review.join(biz).map(lambda x: (x[1][0], x[1][1]))
output = sc.union([header, joined])
#%%
def to_csv(x):
    return ','.join(str(d) for d in x)
output = output.coalesce(1).persist().map(to_csv)
output.saveAsTextFile('data/user_state.csv')
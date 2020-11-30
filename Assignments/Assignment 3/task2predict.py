from __future__ import division
from pyspark import SparkContext
import pickle
from math import sqrt
import sys
import time
import json
#%%
input_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]
#%%
def cosineSimilarity(x):
    user_pr, biz_pr = set(x[0][1]), set(x[1][1])
    similarity = len(user_pr.intersection(biz_pr))/(sqrt(len(user_pr)) * sqrt(len(biz_pr)))
    return {'user_id': x[0][0], 'business_id': x[1][0], 'sim': similarity}
#%%
def contentBasedPrediction(user, biz, input_path):
    user = sc.parallelize(iter(user.items()))
    biz = sc.parallelize(iter(biz.items()))
    test = sc.textFile(input_path).map(json.loads)\
        .map(lambda x: (x['user_id'],x['business_id'])).map(lambda x: (x[1],x[0]))
    joined = test.join(biz).map(lambda x: (x[1][0], (x[0],x[1][1])))
    join2 = joined.join(user).map(lambda x: ((x[0],x[1][1]), x[1][0])).map(cosineSimilarity).filter(lambda x: x['sim'] >= 0.01)
    return join2.collect()
#%%
sc = SparkContext('local[*]', 'task2predict')

start = time.time()

with open(model_path, "rb") as model_file:
    user, biz = pickle.load(model_file)
result = contentBasedPrediction(user, biz, input_path)

with open(output_path, 'w') as f:
    for i in result:
        f.write(json.dumps(i) + '\n')

end = time.time()
print(round(end-start,2))
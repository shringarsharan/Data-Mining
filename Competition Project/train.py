from __future__ import division
import sys, time, json, pickle, binascii, numpy as np, pickle
from pyspark.ml.recommendation import ALS
#%%
from pyspark.sql import SparkSession
import pandas as pd
import json, sys, os
from sys import getsizeof
from datetime import datetime
import xgboost as xgb
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#%%
start = time.time()
input_dir = '../resource/asnlib/publicdata/'
model_path = os.getcwd()+'/model'
als_model_path = os.getcwd()+'/als_model'
#%%
user_avg, item_avg = json.load(open(input_dir+'user_avg.json')), json.load(open(input_dir+'business_avg.json'))
#%%
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
train = sc.textFile(input_dir+'train_review.json').map(json.loads).map(lambda x: (x['user_id'],x['business_id'],x['stars']))
userInt = sc.broadcast(train.keys().distinct().zipWithIndex().collectAsMap())
bizInt = sc.broadcast(train.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap())
train = train.map(lambda x: (userInt.value.get(x[0]), bizInt.value.get(x[1]), x[2])).toDF(['user_id','business_id','stars'])
# Model 1
als_model = ALS(maxIter=20, regParam=0.4, userCol='user_id', itemCol='business_id', ratingCol='stars', coldStartStrategy="nan")
als_model = als_model.fit(train)

del train

#%%
weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
def modifyHours(x):
    if x is not None:
        days = x.keys()
        if all(day in weekdays[:5] for day in days):
            return 'd'
        elif all(day in weekdays[5:] for day in days):
            return 'e'
        else:
            return 'b'
    else:
        return None
def userFeatureEngg(x):
    x.update({'friends':len(x['friends']), 'elite':len(x['elite']), 'user_avg':user_avg.get(x['user_id'],None),
              'yelping_since':2020-datetime.strptime(x['yelping_since'],'%Y-%m-%d %H:%M:%S').year})
    return x
def bizFeatureEngg(x):
    x.update({'categories':x['categories'].split(', '), 'hrs':modifyHours(x['hours']),
              'biz_avg':biz_avg.get(x['business_id'],None)})
    return x
#%%

user_avg, biz_avg = json.load(open(input_dir+'user_avg.json')), json.load(open(input_dir+'business_avg.json'))
train = sc.textFile(input_dir+'train_review.json').map(json.loads).toDF()
user = sc.textFile(input_dir+'user.json').map(json.loads).map(lambda x: userFeatureEngg(x)).toDF()
biz = sc.textFile(input_dir+'business.json').map(json.loads).map(lambda x: bizFeatureEngg(x)).toDF()

train = train.join(biz, on='business_id')
train = train.join(user, on='user_id')

del user, biz
selected_columns = ['stars','user_avg','biz_avg','latitude','longitude','yelping_since','useful','funny',
                    'cool','elite','friends','fans','compliment_hot','compliment_writer',
                    'compliment_photos']

train = train.select(selected_columns)
train = train.toPandas()
scaler = MinMaxScaler()

# Model 2
target = train.pop('stars')
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(train, target, test_size = 0.4)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

print('initialize model')

#model = xgb.XGBRegressor()
#parameters_grid = {'learning_rate' : [0.1, 0.2, 0.5],'max_depth' : [5, 10, 15],
#                   'n_estimators' : [150, 250, 300],'min_child_weight' : [3, 5, 10] }
#cv = model_selection.StratifiedShuffleSplit(n_splits=3, test_size = 0.3, random_state=2)
#rcv = model_selection.RandomizedSearchCV(model, parameters_grid, n_iter=3, scoring ='neg_mean_squared_error',cv=cv)
#mod = rcv.fit(train_data, train_labels)
#xg_model = mod.best_estimator_

xg_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=15, n_estimators=250)

print('fitting')

xg_model.fit(train_data, train_labels)

#%%
print('model done')

models = [scaler, xg_model, userInt.value, bizInt.value]

als_model.write().overwrite().save(als_model_path)
pickle.dump(models, open(model_path, "wb"))

end=time.time()
print(end-start)


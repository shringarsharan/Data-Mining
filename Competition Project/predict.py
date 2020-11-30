from __future__ import division
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os, sys, time, json, pickle, numpy as np
from pyspark.ml.recommendation import ALS, ALSModel
#%%
import pandas as pd
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
test_path = sys.argv[1]
output_path = sys.argv[2]
print(test_path)
#%%
def correctPred(x):
    if x > 5:
        return 5
    elif x < 1:
        return 1
    else:
        return x
#%%
conf = SparkConf().setMaster("local[3]")
spark = SparkSession.builder.config(conf = conf).getOrCreate()
sc = spark.sparkContext
user_avg, biz_avg = json.load(open(input_dir+'user_avg.json')), json.load(open(input_dir+'business_avg.json'))
global_avg = 3.7961611526341503

als_model = ALSModel.load(als_model_path)
with open(model_path, "rb") as files:
    scaler, xg_model, userInt, bizInt = pickle.load(files)

test = sc.textFile(test_path).map(json.loads).map(lambda x: (x['user_id'],x['business_id']))

for user,biz in test.collect():
    if user not in userInt:
        userInt[user] = len(userInt)
    if biz not in bizInt:
        bizInt[biz] = len(bizInt)

test = test.map(lambda x: (userInt.get(x[0],'UNKNOWN'), bizInt.get(x[1],'UNKNOWN'))).toDF(['user_id','business_id'])
#%% Predictions
# ALS Prediction
pred = als_model.transform(test)
pred = pred.withColumnRenamed('prediction','pred_als').toPandas()
pred['pred_als'] = pred['pred_als'].fillna(global_avg).apply(lambda x: correctPred(x))
#%%
userInt, bizInt = {v:k for k,v in userInt.items()}, {v:k for k,v in bizInt.items()}
pred['user_id'] = pred['user_id'].apply(lambda x: userInt[x])
pred['business_id'] = pred['business_id'].apply(lambda x: bizInt[x])
#%%
# XGBoostRegressor Prediction

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
test = sc.textFile(test_path).map(json.loads).toDF()

user = sc.textFile(input_dir+'user.json').map(json.loads).map(lambda x: userFeatureEngg(x)).toDF()
biz = sc.textFile(input_dir+'business.json').map(json.loads).map(lambda x: bizFeatureEngg(x)).toDF()

test = test.join(biz, on='business_id', how='left')
test = test.join(user, on='user_id', how='left')

test_select_columns = ['user_id','business_id','user_avg','biz_avg','latitude','longitude','yelping_since','useful','funny',
                    'cool','elite','friends','fans','compliment_hot','compliment_writer',
                    'compliment_photos']

test = test.select(test_select_columns).toPandas()
user_biz_ids = test[['user_id','business_id']]

test = test.drop(['user_id','business_id'],axis=1)
test = scaler.transform(test)

del user, biz, userInt, bizInt

pred_xg = pd.concat([user_biz_ids, pd.Series(xg_model.predict(test))], axis=1)
pred_xg.columns = ['user_id','business_id','pred_xg']
pred_xg['pred_xg'] = pred_xg['pred_xg'].apply(lambda x: correctPred(x))

pred = pred.merge(pred_xg, on=['user_id','business_id'], how='inner')
del test, pred_xg

pred['stars'] = 0.1*pred['pred_als'] + 0.9*pred['pred_xg']
pred.drop(['pred_als','pred_xg'],axis=1,inplace=True)

pred.to_json(output_path, orient='records', lines=True)

end=time.time()
print(end-start)
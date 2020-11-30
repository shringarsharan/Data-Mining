from pyspark import SparkContext
import json
import codecs
import sys
import itertools
from collections import defaultdict
import os
#%%
review_input_path = sys.argv[1]
biz_input_path = sys.argv[2]
out_file_path = sys.argv[3]
if_spark = sys.argv[4]
num = sys.argv[5]
num = int(num)
#%%
def with_spark(sreview,sbiz,num):
    sreview = sreview.map(lambda x: (x['business_id'], x['stars']))
    sbiz = sbiz.map(lambda x: (x['business_id'], x['categories']))
    joined = sreview.join(sbiz).filter(lambda x: (x[1][0] is not None) & (x[1][1] is not None))
    mid = joined.flatMap(lambda x: [(i.strip(),x[1][0]) for i in x[1][1].split(',')])
    #aTuple = (0,0)
    result = mid.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),lambda a,b: (a[0] + b[0], a[1] + b[1]))\
        .map(lambda x: [x[0],x[1][0]/x[1][1]]).sortBy(lambda x: [-x[1],x[0]]).take(num)
    return result
#%%
def without_spark(review,biz,num):
    biz = list(set(map(lambda x: (x['business_id'], x['categories']),
                       filter(lambda x: x['categories'] is not None, biz))))
    dbiz = dict(biz)
    review = list(filter(lambda x: (x[0] is not None) & (x[1] is not None),
                         map(lambda x: (x['business_id'], x['stars']), review)))
    revfilter = filter(lambda x: x[0] in dbiz.keys(), review)
    med = map(lambda x: (x[0], x[1], dbiz[x[0]]), revfilter)
    result = list(itertools.chain.from_iterable(list(map(lambda x:[(i.strip(), x[1]) for i in x[-1].split(',')], med))))
    finaldict = defaultdict(list)
    for k, v in result:
        finaldict[k].append(v)
    final_result = {k: sum(v)/len(v) for (k, v) in finaldict.items()}
    result = list(map(list, sorted(list(final_result.items()), key=lambda x: [-x[1], x[0]])[:num]))
    return result
#%%
output = {}
if if_spark == "spark":
    sc=SparkContext('local[*]','task2')
    review = sc.textFile(review_input_path).map(json.loads).persist()
    biz = sc.textFile(biz_input_path).map(json.loads).persist()
    output['result'] = with_spark(review,biz,num)
else:
    review = [json.loads(line) for line in codecs.open(review_input_path, 'r', 'utf-8-sig')]
    biz = [json.loads(line) for line in codecs.open(biz_input_path, 'r', 'utf-8-sig')]
    output['result'] = without_spark(review,biz,num)
json_out = json.dumps(output)
with open(out_file_path,"w") as f:
    f.write(json_out)

import os
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf
from itertools import combinations
from graphframes import *
import json, csv
import sys, time

#%%
thresh = int(sys.argv[1])
input_path = sys.argv[2]
output_path = sys.argv[3]
#%%
conf = SparkConf().setMaster("local[3]")
spark = SparkSession.builder.config(conf = conf).getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("OFF")
#sc.setSystemProperty('spark.driver.memory', '4g')
#sc.setSystemProperty('spark.executor.memory', '4g')
#%%
start = time.time()
rdd = sc.textFile(input_path).mapPartitions(csv.reader)
header = rdd.first()
rdd = rdd.filter(lambda x: x != header)
#%% 4.1 Graph Construction

rdd = rdd.map(lambda x: (x[1], x[0])).groupByKey().flatMap(lambda x: [(i,x[0]) for i in list(combinations(sorted(set(x[1])),2))])\
    .groupByKey().mapValues(list).filter(lambda x: len(x[1]) >= thresh)

nodes = rdd.keys().flatMap(list).distinct().map(lambda x: tuple([x])).toDF(["id"])
edges = rdd.keys().flatMap(lambda x: (x,(x[1], x[0]))).toDF(["src", "dst"])
#%%
graph = GraphFrame(nodes, edges)
#%%
communities = graph.labelPropagation(maxIter=5)
communities = communities.rdd.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: sorted(set(x[1]))).sortBy(lambda x: (len(x),x[0])).collect()
#%%
with open(output_path,"w") as f:
    for i in communities:
        out = str(i).replace('[', '').replace(']', '')+"\n"
        f.writelines(out)

end = time.time()

print("Duration:", end-start)

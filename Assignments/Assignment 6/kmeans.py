from pyspark import SparkContext
import time, sys, os, random, math, csv, json
from statistics import mean
from collections import defaultdict
#%%
#os.environ['PYSPARK_PYTHON'] = 'usr/local/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'usr/local/bin/python3.6'
#%%
input_path = sys.argv[1]
n_cluster = int(sys.argv[2])
output_path = sys.argv[3]
#%%
start =time.time()
sc = SparkContext("local[*]", "kmeans")
rdd = sc.textFile(input_path).map(lambda x: x.split(",")).map(lambda x: (int(x[0]), [float(i) for i in x[1:]]))#.map(lambda x: {x[0]: x[1:]})
#%%
def initialize(n_cluster, num_pts):
    random.seed(2)
    return sorted(random.sample(range(num_pts), n_cluster))

def initialize2(n_cluster, data):
    centers = []
    random.seed(2)
    centers.append(random.randint(0,len(data)-1))
    for i in range(1,n_cluster):
        scores = {p: min([eucDistance(data[p],data[C]) for C in centers]) for p in data}
        centers.append(max(scores.items(), key=lambda x:x[1])[0])
    return centers

def eucDistance(P1_dim, P2_dim):
    return math.sqrt(sum((x1-x2)**2 for x1,x2 in zip(P1_dim,P2_dim)))

def assignCluster(cluster_centers, data):
    assigned_cluster = {point: min((eucDistance(data[point],cluster_centers[C]),index) for index,C in
                               enumerate(cluster_centers))[1] for point in data}
    return assigned_cluster

def centroid(grpByCluster, data):
    centroid = [(index,[mean(d) for d in zip(*(data[i] for i in grpByCluster[key]))]) for index,key in enumerate(grpByCluster)]
    return dict(sorted(centroid))
#%% Without rdd
data = rdd.collectAsMap()
maxIter = 100
cluster_centers = dict(sorted((index,data[center]) for index,center in enumerate(initialize2(n_cluster,data))))
assigned_cluster = assignCluster(cluster_centers, data)
grpByCluster = defaultdict(list); [grpByCluster[v].append(k) for k, v in assigned_cluster.items()]
objective = sum(sum(eucDistance(data[i], cluster_centers[C]) for i in points) for C, points in grpByCluster.items())

iter = 1
t = []
t.append(objective)
prevObjective = objective + 1
while objective < prevObjective and iter <= maxIter:
    prevObjective = objective
    cluster_centers = centroid(grpByCluster, data)
    assigned_cluster = assignCluster(cluster_centers, data)
    grpByCluster = defaultdict(list); [grpByCluster[v].append(k) for k, v in assigned_cluster.items()]
    objective = sum(sum(eucDistance(data[i], cluster_centers[C]) for i in points) for C, points in grpByCluster.items())
    t.append(objective)
    iter += 1

assigned_cluster = {str(k):v for k,v in assigned_cluster.items()}
with open(output_path, 'w') as f:
    f.write(json.dumps(assigned_cluster))
end = time.time()
print(end-start)
print(t)
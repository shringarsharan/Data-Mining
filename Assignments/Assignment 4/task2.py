from __future__ import division
from pyspark import SparkContext
from itertools import combinations, chain, permutations
from collections import Counter, defaultdict, deque
from operator import add
import sys
import time
import csv
#%%
case = int(sys.argv[1])
input_path = sys.argv[2]
output_btw_path = sys.argv[3]
output_com_path = sys.argv[4]
#%%
def findingCommunities(test_graph):
    nodes = test_graph.keys()
    comm, vis = [], []
    for i in nodes:
        if i not in vis:
            temp = bfs(i, test_graph)
            comm.append(temp)
            vis.extend(temp)
    return comm
#%%
def calcModularity(communities, orig_graph, all_edges, case):
    m = len(all_edges)/2
    pairs = [list(permutations(sorted(comm),2)) for comm in communities]
    modularity = sum(sum((1 - ((len(orig_graph[a])*len(orig_graph[b]))/(case*m))) if (a,b) in all_edges
                else (0 - ((len(orig_graph[a])*len(orig_graph[b]))/(case*m))) for a,b in comm) for comm in pairs)
    modularity = modularity/(2*m)
    return modularity
#%%
def jaccard(x, user_states):
    U0, U1 = user_states[x[0]], user_states[x[1]]
    jaccard_sim = len(set(U0).intersection(U1))/len(set(U0).union(U1))
    return jaccard_sim
#%%
def girvanNewman(x, nodes_edges):
    # Step 1: BFS
    root = x[0]
    level, child_parent, queue = {root:0}, defaultdict(list), deque([root])
    child_parent[root] = []
    while len(queue) != 0:
        vertex = queue.popleft()
        temp = set(nodes_edges[vertex]).difference(level)
        queue.extend(temp)
        level.update([(i, level[vertex] + 1) for i in temp])
        [child_parent[i].extend([vertex]) for i in nodes_edges[vertex] if level[i] > level[vertex]]
    # Step 2: Labeling each node with number of shortest paths to that node from root
    step2label = {}
    for i in child_parent:
        if i==root:
            step2label[i] = 1
        else:
            step2label[i] = sum(step2label[j] for j in child_parent[i])
    # Step 3: Calculating betweenness of each edge
    edge_betweenness, nodes = Counter(), Counter()
    while child_parent:
        current = child_parent.popitem()
        nodes[current[0]] += 1
        step2wts = [step2label[i] for i in current[1]]
        credit = [(i*nodes[current[0]])/sum(step2wts) for i in step2wts]
        edges = [tuple(sorted((current[0],node))) for node in current[1]]
        for i in range(len(current[1])):
            edge_betweenness[edges[i]] += credit[i]
            nodes[current[1][i]] += credit[i]
    return sorted(edge_betweenness.items())
#%%
def bfs(root, nodes_edges):
    visited, queue = [root], deque([root])
    while len(queue) != 0:
        vertex = queue.popleft()
        temp = set(nodes_edges[vertex]).difference(visited)
        queue.extend(temp), visited.extend(temp)
    return visited
#%%
sc = SparkContext('local[*]', 'task2')
sc.setLogLevel("OFF")
rdd = sc.textFile(input_path).mapPartitions(csv.reader)
header = rdd.first()
rdd = rdd.filter(lambda x: x != header)
start = time.time()

if case==1:
    graph = rdd.groupByKey().mapValues(lambda x: list(set(x))).union(rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list))

if case == 2:
    user_states = dict(rdd.groupByKey().mapValues(lambda x: list(set(x))).collect())
    graph = rdd.map(lambda x: (x[1], x[0])).groupByKey() \
        .flatMap(lambda x: [(i, x[0]) for i in list(combinations(sorted(set(x[1])), 2))]).groupByKey().mapValues(list) \
        .filter(lambda x: jaccard(x[0], user_states) >= 0.5).flatMap(lambda x: (x[0], (x[0][1], x[0][0]))) \
        .map(lambda x: (x[0], x)).groupByKey().mapValues(list).mapValues(lambda x: sorted(i[1] for i in x))

orig_graph, test_graph = dict(graph.collect()), dict(graph.collect())
btw = graph.flatMap(lambda x: girvanNewman(x, test_graph)).reduceByKey(add).map(lambda x: (x[0], x[1] / 2)) \
        .sortBy(lambda x: (-x[1], x[0][0]))
all_edges = btw.keys().flatMap(lambda x: (x, (x[1], x[0]))).collect()
result1 = btw.map(list).collect()
btw = btw.collectAsMap()

communities, oldmodularity = [], -1
while btw:
    n1, n2 = list(btw.keys())[0]
    test_graph[n1].remove(n2), test_graph[n2].remove(n1)
    if n2 not in bfs(n1, test_graph):
        community = findingCommunities(test_graph)
        modularity = calcModularity(community, orig_graph, all_edges, case)
        if modularity > oldmodularity:
            oldmodularity = modularity
            communities = community
    btw = graph.flatMap(lambda x: girvanNewman(x, test_graph)).reduceByKey(add) \
        .mapValues(lambda x: x / 2).collectAsMap()
    btw = dict(sorted(btw.items(), key=lambda x: (-x[1], x[0][0])))
result2 = sorted([sorted(i) for i in communities], key=lambda x: (len(x), x[0]))

with open(output_btw_path,"w") as f:
    for i in result1:
        out = str(i).replace('[', '').replace(']', '')+"\n"
        f.writelines(out)
with open(output_com_path,"w") as f:
    for i in result2:
        out = str(i).replace('[', '').replace(']', '')+"\n"
        f.writelines(out)

end = time.time()
print("Duration:", end-start)
print(oldmodularity)
#result2 = sc.parallelize(communities[max(communities)]).map(lambda x: sorted(x)).sortBy(lambda x: (len(x), x[0])).collect()
#result2 = sc.parallelize(communities).map(lambda x: sorted(x)).sortBy(lambda x: (len(x), x[0])).collect()
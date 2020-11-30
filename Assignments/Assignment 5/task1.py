from pyspark import SparkContext
import json, random, time, sys
import binascii
#%%
input_path = sys.argv[1]
testing_path = sys.argv[2]
output_path = sys.argv[3]
#%%
def hashFuncs(num_hash):
    random.seed(2)
    return {i: [random.randint(0,10000), random.randint(0,10000)] for i in range(num_hash)}

def bloomFilter(x, n, hashes):
    intx = int(binascii.hexlify(x.encode('utf8')),16)
    return ((a * intx + b) % n for a,b in hashes.values())
#%%
sc = SparkContext('local[*]','task1')
start = time.time()
n = 80000000
hashes = hashFuncs(1)
bloom_filter = sc.textFile(input_path).map(json.loads).map(lambda x: (x['name'])).distinct()\
    .flatMap(lambda x: bloomFilter(x,n,hashes)).distinct().collect()
bloom_filter2 = [0]*n
for i in bloom_filter:
    bloom_filter2[i] = 1
del bloom_filter
#%%
result = sc.textFile(testing_path).map(json.loads).map(lambda x: (x['name']), preservesPartitioning=True)\
    .map(lambda x: 'T' if all(bloom_filter2[i] for i in bloomFilter(x,n,hashes)) else 'F',preservesPartitioning=True).collect()
#%%
with open(output_path, 'w') as f:
    f.write(str(result).replace('[','').replace(']','').replace(',','').replace("'",''))

end = time.time()
print(end-start)

test1 = sc.textFile(input_path).map(json.loads).map(lambda x: (x['name'])).distinct().collect()
test2 = sc.textFile(testing_path).map(json.loads).map(lambda x: (x['name'])).filter(lambda x: x in test1)

len(list(filter(lambda x: x=='T', result)))



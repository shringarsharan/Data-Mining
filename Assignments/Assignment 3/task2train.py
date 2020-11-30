from __future__ import division
from pyspark import SparkContext
#from itertools import combinations, islice, chain
from collections import Counter
#from difflib import get_close_matches
import sys
import time
from math import log2
import re
import json
import pickle
#%%
input_path = sys.argv[1]
output_path = sys.argv[2]
stopwords_path = sys.argv[3]
#%%
def clean(text, stopwords):
    text = text.lower()
    text = text.replace("(", " ").replace("["," ").replace(","," ").replace("."," ").replace("!"," ").replace("?"," ")\
        .replace(":"," ").replace(";"," ").replace("]"," ").replace(")"," ").replace('\n','').replace("\\", " ")
    text = re.sub(r"(\$*\d+)", "" ,text)
    text = re.sub(r"(\d+.)", " ", text)
    text = text.replace("can\'t", "can not").replace("won\'t", "will not")
    text = text.replace("\'m", " ").replace("\'re", " ").replace("\'ve", " ").replace("\'s", " ").replace("\'ll"," ").replace("\'d", " ")
    text = re.sub(r"\w+(n\'t).", "", text)
    text = text.strip().split(" ")
    text = [i.strip() for i in text if i not in stopwords]
    return text

def FilterTF(text, vocab):
    text = [vocab[w] for w in text if w in vocab.keys()]
    freq = Counter(text)
    max_freq = freq.most_common(1)[0][1]
    TF = [(word, (freq/max_freq)) for word, freq in freq.items()]
    return TF
#%%
def contentBasedModel(review):
    biz_text  = review.repartition(20).map(lambda x: (x['business_id'],x['text'])).groupByKey().\
        mapValues(lambda x: clean(' '.join(list(set(x))), stopwords))
    # Filtering vocab list to exclude very rare words
    vocab = biz_text.flatMap(lambda x: x[1]).collect()
    threshold = 0.0001*len(vocab)/100
    vocab = Counter(vocab)
    vocab = [k for k,v in vocab.items() if v>=threshold]
    vocab = dict([(i1,i0) for i0,i1 in enumerate(vocab)])
    # Total number of documents/businesses
    num_biz = biz_text.count()
    # Creating biz profiles using 200 significant words with highest TF.IDF
    tfidf = biz_text.mapValues(lambda text: FilterTF(text, vocab)).flatMap(lambda x: [(word, (x[0], tf)) for word, tf in x[1]])\
        .groupByKey().mapValues(list).mapValues(lambda x: [(biz_id, tf*log2(num_biz/len(x))) for biz_id,tf in x])
    biz_profile = tfidf.flatMap(lambda x: ([(biz_id, (x[0], tfidf_score)) for biz_id, tfidf_score in x[1]])).groupByKey()\
        .mapValues(lambda x: list(dict(sorted(x, reverse=True, key=lambda x: x[1])[:200]).keys()))
    # Creating user profiles by aggregating significant words for businesses that the user rated
    user = review.map(lambda x: (x['business_id'], x['user_id']))
    user_profile = biz_profile.join(user).map(lambda x: (x[1][1], x[1][0])).groupByKey()\
        .mapValues(lambda x: list(set([item for sublist in x for item in sublist])))
    return user_profile.collect(), biz_profile.collect()
#%%
sc = SparkContext('local[*]', 'task2train')

start = time.time()

stp = list(open(stopwords_path,'r'))
stopwords = [i.replace('\n','') for i in stp]
stopwords.extend(['',' ', "it\'s", "i\'m", "", '&', '$'])
review = sc.textFile(input_path).map(json.loads)

user_profile, biz_profile = contentBasedModel(review)
outcome = [dict(user_profile), dict(biz_profile)]
pickle.dump(outcome, open(output_path, "wb"))

end = time.time()
print(round(end-start,2))
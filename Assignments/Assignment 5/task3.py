import tweepy, json, sys, string, random, csv
from collections import Counter
#%%
output_path = sys.argv[2]
#%%
def checkifascii(text):
    for i in text:
        if i not in string.ascii_letters:
            return False
    return True
#%%
ACCESS_TOKEN = '1205346843237859328-NTCf9d6raWfKmVvG6wWGBfC8tIyikF'
ACCESS_TOKEN_SECRET = 'BMzHnlSwghMcPTFZ7NvB0X4vxBWA7Z59UCmo3SylNYRRI'
CONSUMER_KEY = 'vbIrbCfbfz56lSSpdpXGCiPxz'
CONSUMER_SECRET = 'srxCrNb32alUiXGZ0PH7sX4ned3HvvlayaZaNHPmyYjKe9A57Z'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth_handler=auth)

class twitterListener(tweepy.StreamListener):
    # def __init__(self, output_path):
    #     self.output_path = output_path

    def on_status(self, status):
        global seq_num
        global tweet_sample
        taglist = status.entities['hashtags']

        if taglist:
            for tag in taglist:
                if checkifascii(tag['text']):
                    seq_num += 1
                    if seq_num < 100:
                        tweet_sample.append(tag['text'])
                    else:
                        random_replace = random.randint(0,seq_num)
                        if random_replace < 100:
                            tweet_sample[random_replace] = tag['text']
                    frequency = Counter(tweet_sample)
                    top3 = {i: frequency[i] for i in frequency if frequency[i] in
                            sorted(set(frequency.values()), reverse = True)[:3]}
                    top3 = {k: v for k, v in sorted(top3.items(), key=lambda x: (-x[1],x[0]))}
                    print(top3)
                    with open(output_path, 'a+') as f:
                        writer = csv.writer(f)
                        writer.writerow([f"The number of tweets with tags from the beginning: {seq_num}"])
                        for k,v in top3.items():
                            writer.writerow([f"{k}: {v}"])

    def on_error(self, status_code):
        if status_code == 420:
            return False
#%%
listener = twitterListener()
stream = tweepy.Stream(auth=api.auth, listener=listener)
#%%
seq_num = 0
tweet_sample = []
stream.sample(languages=['en'])

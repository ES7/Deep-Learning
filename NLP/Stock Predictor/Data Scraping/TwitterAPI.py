import tweepy
import pandas as pd
import time

api_key = ""
api_key_secret = ""

access_token = ''
access_token_secret = ''

bearer_token = ''

keyword = 'APPL stocks'
# authentication
# client = tweepy.Client(bearer_token, api_key, api_key_secret, access_token, access_token_secret)
auth = tweepy.OAuthHandler(api_key, api_key_secret)
# auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
#-------------------------------------------------------------------------------------------------------------------------------------
cursor = tweepy.Cursor(api.search_tweets, q=keyword, count=200,tweet_mode='extended').items(1000)
data = []
columns = ['Time', 'User', 'Tweet']
for tweet in cursor:
    data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)

df.to_csv('APPL.csv')
# print(df)
#-------------------------------------------------------------------------------------------------------------------------------------
# class Listener(tweepy.Stream):
#     def on_status(self,status):
#         print(status.user.screen_name + ":" + status).text


# stream = Listener(api_key, api_key_secret, access_token, access_token_secret) 

# try:
#     stream.filter(track=keyword)
# except Exception as e:
#     print(e)
#-------------------------------------------------------------------------------------------------------------------------------------
# class myStream(tweepy.StreamingClient):
#     def on_closed(self):
#         print("COnnected")
    
#     def on_tweet(self, tweet):
#         if tweet.referenced_tweets == None:
#             print(tweet.text)
#             time.sleep(0.2)

# stream = myStream(bearer_token)
# stream.filter()
#-------------------------------------------------------------------------------------------------------------------------------------
# public_tweets = api.search_tweets(q=keyword, tweet_mode='extended')

# # create dataframe
# columns = ['Time', 'User', 'Tweet']
# data = []
# for tweet in public_tweets:
#     data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])

# df = pd.DataFrame(data, columns=columns)

# df.to_csv('tweets.csv')

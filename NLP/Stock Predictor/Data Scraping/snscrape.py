import snscrape.modules.twitter as sntwitter
import pandas as pd

inp = input("Enter the stock name : ")
keyword = f'{inp} until:2020-01-01 since:2015-01-01'
nos = input("Number of Tweets you want : ")
tweets = []

for tweet in sntwitter.TwitterSearchScraper(keyword).get_items():
    if len(tweets) == int(nos):
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
print(df['Tweet'].dtype)
df.to_csv(f'{inp}_tweets.csv')

# import os, sys
# os.execv('Streaming_try_2.py',sys.argv)

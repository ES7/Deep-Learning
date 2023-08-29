# 1/10 Predictor
In the `Data Scraping` folder I have wrote code for scraping Twitter comments using `TwitterAPI` and `snscrape` library. Inside the `Models` folder, in the `BERT.ipynb` I have used `BERT` model for sentiment analysis and applied it to all the scraped comments of a particular stock. And using the sentiment score the code predicts whether the stocks goes UP or DOWN. 
## BERT Model
Here I have used **"nlptown/bert-base-multilingual-uncased-sentiment"** model to perform the task. I have loaded this pre-trained model with its tokenizer and apply it on the dataset to get the sentiments of the comments.</br>
The sentiments are as (1,2,3,4,5) that is very negative, negative, neutral, positive, very positive respectively. </br>
Using these sentiment values I have implemented the logic for stock prediction as :-
* If the sentiment is 1 and 2 then the stocks goes down.
* If the sentiment is 4 and 5 then the stocks goes up.
* If the sentiment is 3 then the stocks stays neutral.

## LSTM Model
Here the dataset is in .csv format which I have downloaded form kaggle https://www.kaggle.com/datasets/hershyandrew/amzn-dpz-btc-ntfx-adjusted-may-2013may2019 </br>
After getting the data I have cleaned it using the standard methods of ML while handling .csv data. Then I have perform visualization of some features of the dataset using matplotlib and seaborn :- 
* **Closing Price :** The closing price is the last price at which the stock is traded during the regular trading day. A stock’s closing price is the standard benchmark used by investors to track its performance over time.
* **Volume of Sales :** Volume is the amount of an asset or security that changes hands over some period of time, often over the course of a day. For instance, the stock trading volume would refer to the number of shares of security traded between its daily open and close. Trading volume, and changes to volume over the course of time, are important inputs for technical traders. </br>
* **Moving average of stocks :** (MA) is a simple technical analysis tool that smooths out price data by creating a constantly updated average price. The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks, or any time period the trader chooses.
* **Correlation between different stocks closing prices :** Correlation is a statistic that measures the degree to which two variables move in relation to each other which has a value that must fall between -1.0 and +1.0. Correlation measures association, but doesn’t show if x causes y or vice versa — or if the association is caused by a third factor.
At last I have perform the prediction of the closing price of APPLE stocks using LSTM.

# 1/10 Predictor
In the `Data Scraping` folder I have wrote code for scraping Twitter comments using `TwitterAPI` and `snscrape` library. Inside the `Models` folder, in the `BERT.ipynb` I have used `BERT` model for sentiment analysis and applied it to all the scraped comments of a particular stock. And using the sentiment score the code predicts whether the stocks goes UP or DOWN. 
## BERT Model
Here I have used **"nlptown/bert-base-multilingual-uncased-sentiment"** model to perform the task. I have loaded this pre-trained model with its tokenizer and apply it on the dataset to get the sentiments of the comments.</br>
The sentiments are as (1,2,3,4,5) that is very negative, negative, neutral, positive, very positive respectively. </br>
Using these sentiment values I have implemented the logic for stock prediction as :-
* If the sentiment is 1 and 2 then the stocks goes down.
* If the sentiment is 4 and 5 then the stocks goes up.
* If the sentiment is 3 then the stocks stays neutral.

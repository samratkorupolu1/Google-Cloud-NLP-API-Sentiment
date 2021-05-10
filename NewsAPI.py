#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install newsapi-python
#https://newsapi.org/docs/endpoints/everything

from newsapi import NewsApiClient #importing the newsapi library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from string import punctuation
from nltk.corpus import stopwords
import re
import nltk
import numpy as np


# In[3]:


#Running the python script containing API keys
get_ipython().run_line_magic('run', 'C:\\\\Users\\\\korupos\\\\Documents\\\\Sentiment\\\\key.ipynb')


# In[4]:


x = "desantis"


# In[5]:


#Initialize API
newsapi = NewsApiClient(api_key= news_api)

# /v2/sources
#sources = newsapi.get_sources()

# /v2/top-headlines
#top_headlines = newsapi.get_top_headlines(q='florida + vaccine',language='en', country = 'us')

# /v2/everything
all_articles = newsapi.get_everything(q= x, from_param='2021-04-11',
                                      to='2021-05-10', language='en', sort_by='relevancy', page=1, page_size = 100)


# In[6]:


print(all_articles.keys())
print('Total articles : ' + str(all_articles['totalResults']))


# In[7]:


all_articles


# In[9]:


#List of news sources 

# print('Total sources :' + str(len(sources['sources'])))
# print()
# for i in range(len(sources['sources'])):
#     print(i+1, sources['sources'][i]['name'])


# In[10]:


news_df = pd.DataFrame(all_articles['articles'])


# In[14]:


news_df[:1]


# ## GCP NLP API

# In[15]:


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "C:\\Users\\korupos\\Documents\\Sentiment\\senti-311518-1b03476e63cd.json"


# In[16]:


# Imports the Google Cloud client library
from google.cloud import language_v1


# In[17]:


# Instantiates a client
client = language_v1.LanguageServiceClient()


# In[18]:


senti = []
for i in news_df['title']:
    text = i
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
    senti.append(round(sentiment.score, 2))


# In[19]:


news_df['sentiment_score'] = senti


# In[20]:


news_df.to_csv('newsAPI.csv',index=False,encoding='utf-8')


# In[22]:


pos = news_df[news_df.sentiment_score > 0]
neg = news_df[news_df.sentiment_score < 0]
neu = news_df[news_df.sentiment_score == 0]


# ## Results

# In[23]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Pos', 'Neg', 'Neu'
sizes = [len(pos), len(neg), len(neu)]
colors = 'green','brown','grey'

explode = (0, 0.05, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(20,10))
_, _, autopcts = ax1.pie(sizes, explode=explode, labels=labels, colors = colors, autopct='%1.1f%%', shadow=True, startangle=90,
       textprops={'fontsize': 20})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':48})
ax1.set_title('Sentiment', fontdict={'fontsize': 30})

plt.show()


# In[24]:


pos.sort_values(by=['sentiment_score'], inplace=True, ascending = False)
neg.sort_values(by=['sentiment_score'], inplace=True)


# In[26]:


pos[['title', 'sentiment_score']].head(10)


# In[27]:


neg[['title', 'sentiment_score']].head(10)


# ## News distribution

# In[30]:


print('Model 1 : GCP')
news_df['sentiment_score'].hist(bins=20)
plt.title('Sentiment Histogram')
plt.xlabel('Sentiment scores by GCP NLP')
plt.show()
print('Total Tweets : ' + str(len(news_df)))
print('Neutral Tweets : ' + str((news_df['sentiment_score'] == 0).sum()))
print('Positive Tweets : ' + str((news_df['sentiment_score'] > 0).sum()))
print('Negative Tweets : ' + str((news_df['sentiment_score'] < 0).sum()))


# # Ignore the code below

# ## Cleaning

# In[70]:


# blobs = []                #This list holds the cleaned tweets

# for i in range(len(news_df)):
#     title = news_df['title'][i]
#     title = title.strip(punctuation).lower()
#     title = re.sub(r'http\S+', ' ', title)
#     title = re.sub(r'@\S+', ' ', title)
#     title = re.sub(r'#', '', title)
#     title = re.sub(r'[^a-z]+', ' ', title)
#     title = re.sub(r'rt', ' ', title)
    
#     blobs.append(title)


# ## Model 1 : TextBlob

# In[71]:


# polarity = []
# subjectivity = []

# for i in range(len(blobs)):
#     xyz = TextBlob(blobs[i])
#     polarity.append(xyz.sentiment.polarity)
#     subjectivity.append(xyz.sentiment.subjectivity)


# In[72]:


# news_df['Polarity'] = polarity
# news_df['subjectivity'] = subjectivity


# In[74]:


# news_df['Polarity'].mean()


# ## Model 2 : Vader

# In[80]:


# analyzer = SentimentIntensityAnalyzer()
# sentiment = {}
# pol = [0]*len(news_df)
    
# for i in range(len(news_df)):
#     title = news_df['title'][i]
#     sentiment[i] = analyzer.polarity_scores(title)
#     pol[i] = sentiment[i]['compound']
# news_df['Polarity_1'] = pol


# ## Results

# In[77]:


# print('Model 1 : TextBlob')
# news_df['Polarity'].hist(bins=20)
# plt.title('Sentiment Histogram')
# plt.xlabel('Sentiment Polarity Model 1')
# plt.show()
# print('Total Titles : ' + str(len(news_df)))
# print('Neutral Titles : ' + str((news_df['Polarity'] == 0).sum()))
# print('Positive Titles : ' + str((news_df['Polarity'] > 0).sum()))
# print('Negative Titles : ' + str((news_df['Polarity'] < 0).sum()))


# In[81]:


# print('Model 2 : Vader')
# news_df['Polarity_1'].hist(bins=20)
# plt.title('Sentiment Histogram')
# plt.xlabel('Sentiment Polarity Model 2')
# plt.show()
# print('Total Titles : ' + str(len(news_df)))
# print('Neutral Titles : ' + str((news_df['Polarity_1'] == 0).sum()))
# print('Positive Titles : ' + str((news_df['Polarity_1'] > 0).sum()))
# print('Negative Titles : ' + str((news_df['Polarity_1'] < 0).sum()))


# In[84]:





# In[92]:


# for i in range(len(news_df)):
#     if news_df['Polarity'][i] > 0:
#         print(news_df['title'][i])


# In[ ]:





# In[ ]:





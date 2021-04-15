#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[2]:


page = requests.get("https://www.sayfiereview.com/")


# In[3]:


soup = BeautifulSoup(page.content, 'html.parser')


# In[4]:


div_bcd = soup.find('div', class_ ='blogticker_content_div')
len(div_bcd.find_all('a', class_='daily_link_link'))


# In[5]:


soup.find_all('a', class_='daily_link_link')
len(soup.find_all('a', class_='daily_link_link'))


# In[14]:


headline = []
for i in range(len(soup.find_all('a', class_='daily_link_link'))):
    headline.append((soup.find_all('a', class_='daily_link_link')[i].text).split(':', 1))
for i in range(len(div_bcd.find_all('a', class_='daily_link_link'))):
    headline.append((div_bcd.find_all('a', class_='daily_link_link')[i].text).split(':', 1))


# In[15]:


df = pd.DataFrame(headline,columns=['source','headline'])


# In[27]:


#DROPPED FIRST ROW AS IT'S HEADLINE IS NULL
df = df.drop(df.index[0])


# In[17]:


#import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "C:\\Users\\korupos\\Documents\\FLShots\\sentiment-310619-8cf4e49950c0.json"
#https://github.com/samratkorupolu1/Sentiment_FLNEWS_GCNLP/blob/main/sentiment-310619-8cf4e49950c0.json


# In[6]:

path = "C:\\Users\\carlsod\\AppData\\Local\\Temp\\Engine_14584_aaa14cb3be8d43fd8a4a81761363e67e_\\818c08a7-b06f-48f3-bef2-af6ff7eda262\\Sentiment_FLNEWS_GCNLP\\sentiment-310619-8cf4e49950c0.json"
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= path

# In[7]:


# Imports the Google Cloud client library
from google.cloud import language_v1


# In[8]:


# Instantiates a client
client = language_v1.LanguageServiceClient()


# In[42]:


senti = []
for i in df['headline']:
    text = i
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
    senti.append(round(sentiment.score, 2))


# In[51]:


df['sentiment_score'] = senti


# In[112]:


df.to_csv('sayfiereview.csv')


# In[48]:


df['sentiment score'].hist(bins=20)


# In[65]:


pos = df[df.sentiment_score > 0]
neg = df[df.sentiment_score < 0]
neu = df[df.sentiment_score == 0]


# In[90]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Pos', 'Neg', 'Neu'
sizes = [len(pos), len(neg), len(neu)]
colors = 'green','brown','grey'

explode = (0, 0.05, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(24,12))
_, _, autopcts = ax1.pie(sizes, explode=explode, labels=labels, colors = colors, autopct='%1.1f%%', shadow=True, startangle=90,
       textprops={'fontsize': 28})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':48})
ax1.set_title('Sentiment', fontdict={'fontsize': 37})

plt.show()


# In[94]:


pos.sort_values(by=['sentiment_score'], inplace=True, ascending = False)
neg.sort_values(by=['sentiment_score'], inplace=True)


# In[110]:


pos.head(10)


# In[111]:


neg.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





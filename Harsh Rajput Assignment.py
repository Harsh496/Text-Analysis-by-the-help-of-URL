#!/usr/bin/env python
# coding: utf-8

# In[1]:


# first we have to import pandas to analyze the data
import pandas as pd


# In[2]:


df = pd.read_excel('input.xlsx')


# In[3]:


df.head(10)


# In[4]:


pip install newspaper3k


# In[5]:


#now we by the help of url we will takeout the text content
from newspaper import Article
import nltk


# In[6]:


url = 'https://insights.blackcoffer.com/online-gaming-adolescent-online-gaming-effects-demotivated-depression-musculoskeletal-and-psychosomatic-symptoms/'
arti = Article(url, language="en")


# In[7]:


import nltk
nltk.download('punkt')


# In[8]:


arti.download() 
arti.parse() 
arti.nlp() 


# In[9]:


print("Article Title:") 
print(arti.title) 
print("\n") 
print("Article Text:") 
print(arti.text) 
print("\n") 
print("Article Summary:") 
print(arti.summary) 
print("\n") 
print("Article Keywords:")
print(arti.keywords) 


# In[10]:


def get_text(url):
    
    url1 = url
    arti = Article(url1, language="en")
    
    arti.download() 
    arti.parse() 
    arti.nlp()
    
    return arti.text


# In[11]:


for i in range(0,len(df)):
    df['URL_ID'][i] = get_text(df['URL'][i])


# In[12]:


df.rename({'URL_ID':'Text'},axis=1,inplace=True)


# In[13]:


df.head(10)


# In[14]:


#now we will have to prepare the text and clean for out dataset
import re
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[15]:


def transform(text):

    review = re.sub('[^a-zA-Z0-9]', ' ',text)  # except small and capital letters and numeric remove everythong.
    review = review.lower()                    # lower it.
    review = review.split()
    
    review = [word for word in review if not word in stopwords.words('english')]   # remove stopwords.
    review = ' '.join(review)
    return review


df['Transform_Text'] = df['Text'].apply(transform)


# In[16]:


#data analysis we have to perform textual and compute varaiable
df['word_counts'] = df['Transform_Text'].apply(lambda x: len(str(x).split()))


# In[17]:


import nltk


# In[18]:


len(nltk.sent_tokenize(df['Text'][0]))  


# In[19]:


import numpy as np


# In[20]:


df['average number of words per sentence'] = np.nan

for i in range(0,len(df)):
    
    df['average number of words per sentence'][i] = df['word_counts'][i]/len(nltk.sent_tokenize(df['Text'][i]))


# In[21]:


df.head(10)


# In[22]:


#average word length 
def char_count(x):
    s = x.split()
    x = ''.join(s)
    return len(x)     


# In[23]:


df['chara_count'] = df['Transform_Text'].apply(lambda x: char_count(x))


# In[24]:


df['average word length'] = np.nan

for i in range(0,len(df)):
    
    df['average word length'][i] = df['chara_count'][i]/df['word_counts'][i]


# In[25]:


df.head()


# In[26]:


#syllable count
R = df.head()


# In[27]:


def syllable_count(x):
    v = []
    d = {}
    for i in x:
        if i in "aeiou":
            v.append(i)
            d[i] = d.get(i,0)+1     
            
    k = []
    for i in d:
        k.append(d[i])
    print(d)
    print(v)  
    print(k)
    print(np.sum(k))
        
    
g = 'bore i am gone to london in england britian uk'

syllable_count(g)


# In[28]:


def syllable_count(x):
    v = []
    d = {}
    for i in x:
        if i in "aeiou":
            v.append(i)
            d[i] = d.get(i,0)+1
            
    k = []
    for i in d:
        k.append(d[i])

    return np.sum(k)

g = R['Transform_Text'][1]

syllable_count(g)


# In[29]:


df['syllable count'] = df['Transform_Text'].apply(lambda x: syllable_count(x))


# In[30]:


df.head()


# In[31]:


# complex word count

from collections import  Counter

def complex_word_count(x):
    
    syllable = 'aeiou'
    
    t = x.split()
    
    v = []
    
    for i in t:
        words = i.split()
        c=Counter()
        
        for word in words:
            c.update(set(word))

        n = 0
        for a in c.most_common():
            if a[0] in syllable:
                if a[1] >= 2:
                    n += 1
                
        m = 0
        p = []
        for a in c.most_common():
            if a[0] in syllable:
                p.append(a[0])
        if len(p) >= 2:
            m += 1
        
        if n >= 1 or m >= 1:
            v.append(i)
            
    return len(v) 

g = R['Transform_Text'][1]

complex_word_count(g)


# In[32]:


df['complex_count'] = np.nan

df['complex_count'] = df['Transform_Text'].apply(lambda x: complex_word_count(x))
df.head()


# In[33]:


#Analysis of Readability
df['sentence length'] = np.nan
df['Average Sentence Length'] = np.nan
df['Percentage of Complex words'] = np.nan
df['Fog Index'] = np.nan


for i in range(0,len(df)):
    
    df['sentence length'][i]  =   len(nltk.sent_tokenize(df['Text'][i]))
    df['Average Sentence Length'][i] = df['word_counts'][i]/df['sentence length'][i]
    df['Percentage of Complex words'][i] = df['complex_count'][i]/df['word_counts'][i] 
    df['Fog Index'][i] = 0.4 * (df['Average Sentence Length'][i] + df['Percentage of Complex words'][i])


# In[34]:


df.head()


# In[35]:


pip install xlrd==1.2.0


# In[36]:


#SENTIMENT ANALYSIS
senti = pd.read_csv('Masterdict.csv')


# In[37]:


dfs = senti[['words','Negative','positive']]
dfs


# In[38]:


f = ['accurate','accidental','bait','accomplishment','accusation']

negative = 0
positive = 0

for i in dfs['words']:
    if i in f:
        if dfs[dfs['words']==i].Negative.any() == True:
            negative += 1
        if dfs[dfs['words']==i].positive.any() == True:               
            positive += 1
            
print(negative),
print(positive)


# In[39]:


# we utilise dfs for sentimental score to  must decrease the word column
dfs = dfs.dropna()
dfs.isnull().sum()


# In[40]:


w = 'the good girl'
w.split()


# In[41]:


dfs['word_lower'] = np.nan


# In[42]:


import warnings
warnings.filterwarnings('ignore')

for i in range(len(dfs)):
        dfs['word_lower'][i] = dfs['words'][i].lower()


# In[43]:


for i in range(50742,len(dfs)):
        dfs['word_lower'][i] = dfs['words'][i].lower()


# In[44]:


dfs['word_lower'].dtype


# In[45]:


dfs.head()


# In[46]:


#positive score 
def positive(x):
    
    s = x.split()
    
    positive = 0
    
    for i in dfs['word_lower']:
        if i in s:
            if dfs[dfs['word_lower']==i].positive.any() == True:
                positive += 1
            
    return positive


# In[47]:


df['positive_score'] = np.nan

for i in range(len(df)):
    df['positive_score'][i] = positive(df['Transform_Text'][i])


# In[48]:


df.head()


# In[49]:


def positive_word(x):
    
    s = x.split()
    
    positive_word = []
    
    for i in dfs['word_lower']:
        if i in s:
            if dfs[dfs['word_lower']==i].positive.any() == True:   
                positive_word.append(i)
            
    print(positive_word)


# In[50]:


df['positive_word'] = np.nan

for i in range(1):
    df['positive_word'][i] = positive_word(df['Transform_Text'][i])


# In[51]:


df.drop('positive_word',axis=1,inplace=True)


# In[52]:


#negative score
def negative_score(x):
    
    s = x.split()
    
    negative = 0
    
    for i in dfs['word_lower']:
        if i in s:
            if dfs[dfs['word_lower']==i].Negative.any() == True:
                negative += 1
            
    return negative


# In[53]:


df['negative_score'] = np.nan

for i in range(len(df)):
    df['negative_score'][i] = negative_score(df['Transform_Text'][i])


# In[54]:


df.head()


# In[55]:


#polarity score
df['Polarity Score'] = np.nan

for i in range(len(df)):
    df['Polarity Score'][i] = (df['positive_score'][i]-df['negative_score'][i])/ ((df['positive_score'][i] + df['negative_score'][i]) + 0.000001)


# In[56]:


df.head()


# In[57]:


pip install TextBlob


# In[58]:


#subjectivity score
from textblob import TextBlob


# In[59]:


blob = TextBlob(df['Transform_Text'][1])
blob.sentiment


# In[60]:


TextBlob(df['Transform_Text'][1]).sentiment[1]


# In[61]:


df['subjectivity'] = np.nan

for i in range(len(df)):
    df['subjectivity'][i] = TextBlob(df['Transform_Text'][i]).sentiment[1]


# In[62]:


df.head()


# In[63]:


pip install spacy


# In[64]:


#personal pronouns
import spacy
from spacy.lang.en.examples import sentences 


# In[65]:


from spacy.cli import download
download ('en_core_web_sm')


# In[66]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[67]:


x = 'she is the my mother'
y = nlp(x)

for noun in y.noun_chunks:
    print(noun)


# In[68]:


doc = nlp('she is the my mother')


# In[69]:


for token in doc:
    if token.pos_ == 'PRON':
        print(token)


# In[70]:


df['PERSONAL PRONOUNS'] = np.nan


# In[71]:


doc = nlp(df['Text'][1])
tok = []
for token in doc:
    if token.pos_ == 'PRON':
        tok.append(token)
        
tok


# In[72]:


df['PERSONAL PRONOUNS'][1] = tok


# In[73]:


df.head()


# In[74]:


df['PERSONAL PRONOUNS'] = np.nan

for i in range(len(df)):
    doc = nlp(df['Text'][i])
    tok = []
    for token in doc:
        if token.pos_ == 'PRON':
            tok.append(token)
        
    df['PERSONAL PRONOUNS'][i] = tok


# In[75]:


df.head()


# In[76]:


df['PERSONAL PRONOUNS'][2]


# In[77]:


submit = df[['URL','positive_score','negative_score','Polarity Score','subjectivity','Average Sentence Length','Percentage of Complex words',
            'Fog Index','average number of words per sentence','complex_count','word_counts','syllable count','PERSONAL PRONOUNS','average word length']]


# In[78]:


submit.head()


# In[79]:


file_name = "Output Data Structure.xlsx"

submit.to_excel(file_name)


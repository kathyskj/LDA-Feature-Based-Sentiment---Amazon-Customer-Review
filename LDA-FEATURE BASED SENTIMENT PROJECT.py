#!/usr/bin/env python
# coding: utf-8

# PREPARING DATA

# In[1]:


import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen

import random
import numpy as np
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict


# In[2]:


get_ipython().system('wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz')


# In[3]:


#Load the meta data

data = []
with gzip.open('meta_Grocery_and_Gourmet_Food.json.gz', 'r') as f:
    for l in tqdm(f):
        data.append(json.loads(l))


# In[4]:


# Convert metadata list into pandas dataframe 

df = pd.DataFrame.from_dict(data)
df.head()


# In[5]:


#Extracting coffee from TITLE column

metadata = df[df["title"].str.contains("coffee|Coffee|Coffees|coffees|COFFEE|COFFEES|kopi|KOPI|Kopi|Café|Kaffee|Kofe|Кофе|Koffie|Kahvi|Kafés|Καφές|Kofee|Kaffi|Cafea|Kaffe")]

metadata.shape


# **Merge Metadata with Review File **

# In[6]:


get_ipython().system('wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Grocery_and_Gourmet_Food.json.gz')


# In[7]:


#Load review data

review = []
with gzip.open('Grocery_and_Gourmet_Food.json.gz', 'r') as f:
    for l in tqdm(f):
        review.append(json.loads(l))


# In[8]:


# convert list into pandas dataframe

df_review= pd.DataFrame.from_dict(review)
df_review.head()


# In[9]:


#Merge both files

coffee=pd.merge(metadata, df_review,on='asin',how='left')

coffee.head()


# In[10]:


coffee.shape


# In[11]:


coffee['category'] = coffee['category'].astype(str).str.replace(r'\[|\]|,', '')
coffee['description'] = coffee['description'].astype(str).str.replace(r'\[|\]|,', '')
coffee['feature'] = coffee['feature'].astype(str).str.replace(r'\[|\]|,', '')
coffee['rank'] = coffee['rank'].astype(str).str.replace(r'\[|\]|,', '')

coffee.head()


# In[12]:


#Select GROCERY from main_Category 

coffee1 = coffee[coffee["main_cat"].str.contains("Grocery")]

coffee1.shape


# In[13]:


#Extracting BEVERAGE from category

coffee2 = coffee1[coffee1["category"].str.contains("Beverages|beverage|beverages|Beverage")]

coffee2.shape


# In[14]:


#Check missing values

coffee2.isnull().sum()


# In[15]:


#Dropping variables: tech1, tech2, also_buy, also_view, similar_item, imageURL, imageURLHighRes, details, rank,price

coffee3=coffee2.drop(['category','tech1','fit', 'tech2', 'also_buy', 'also_view','main_cat', 'similar_item', 'imageURL', 'imageURLHighRes','details','verified','reviewerName','unixReviewTime','vote','image','style','rank','price'], axis=1)

coffee3.shape


# In[16]:


# Drop missing values from row

coffee4=coffee3.dropna(subset=['overall','reviewTime','reviewerID','reviewText','summary'])

coffee4.shape


# In[17]:


#Check missing values

coffee4.isnull().sum()


# In[18]:


# Conconcateneate reviewtext and summary

coffee4['review_text'] = coffee4[['summary', 'reviewText']].apply(lambda x: " ".join(str(y) for y in x if str(y) != 'nan'), axis = 1)
coffee5 = coffee4.drop(['reviewText', 'summary'], axis = 1)

coffee5.head()


# In[19]:


# change column name 
coffee5 = coffee5.rename(columns={'overall': 'Rating'})

print ("Total data:", str(coffee5.shape))
coffee5.head()


# In[20]:


# Convert time object to datetime and create a new column named 'time'#

coffee5['time'] = coffee5.reviewTime.str.replace(',', "")
coffee5['time'] = pd.to_datetime(coffee5['time'], format = '%m %d %Y')

# Drop redundant 'reviewTime' column

coffee6 = coffee5.drop('reviewTime', axis = 1)

coffee6.head()


# In[21]:


# Create new columns

coffee6['day'] = coffee6['time'].dt.day
coffee6['month'] = coffee6['time'].dt.month
coffee6['year'] = coffee6['time'].dt.year

coffee6.head()


# In[22]:


# Classify ratings as good

good_rate = len(coffee6[coffee6['Rating'] >= 3])
bad_rate = len(coffee6[coffee6['Rating'] < 3])

# Printing rates and their total numbers
print ('Good ratings : {} reviews for coffee products'.format(good_rate))
print ('Bad ratings : {} reviews for coffee products'.format(bad_rate))


# In[23]:


# Apply the new classification to the ratings column##

coffee6['rating_class'] = coffee6['Rating'].apply(lambda x: 'bad' if x < 3 else'good')
coffee6.head()


# In[24]:


coffee6.shape


# In[25]:


#DESCRIPTIVE STATISTICS

print ("================================================")

# Total reviews
total = len(coffee6)
print ("Number of reviews: ",total)
print ()

# How many unique products?
print ("Number of unique products: ", len(coffee6.asin.unique()))
product_prop = float(len(coffee6.asin.unique())/total)
print ("Prop of unique products: ",round(product_prop,3))
print ()

# How many unique brands?
print ("Number of unique brands: ", len(coffee6.brand.unique()))
product_prop = float(len(coffee6.brand.unique())/total)
print ("Prop of unique brands: ",round(product_prop,3))
print ()

# Average star score
print ("Average rating score: ",round(coffee6.Rating.mean(),3))


print ("================================================")


# In[26]:


coffee6.to_csv(r'C:\Users\Khadijah Jasni\Desktop\rawcoffee_review.csv')


# In[27]:


get_ipython().system('pip install plotly')


# In[28]:


get_ipython().system('pip install cufflinks')


# In[29]:


import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
values = [(good_rate),(bad_rate)]

ax.pie(values, 
       labels = ['Penilaian Baik', 'Penilaian Buruk'],
       colors=['blue', 'orangered'],
       shadow=False,
       startangle=90, 
       autopct='%1.2f%%')
ax.axis('equal')
plt.title('Produk Kopi: Penilaian Baik Vs. Penilaian Buruk ');

# save the figure
plt.savefig('plot1.png', dpi=300, bbox_inches='tight')


# In[31]:


## PLOT DISTRIBUTION OF RATING 
##########################################

plt.figure(figsize=(12,8))
# sns.countplot(df['Rating'])
coffee6['Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Pengedaran Penilaian Keseluruhan Produk')
plt.xlabel('Penilaian')
plt.ylabel('Bilangan Ulasan')


# In[32]:


#DISTRIBUTION OF RATING SCORE
class_counts = coffee6.groupby('Rating').size()
class_counts


# In[33]:


# Customer totals for each rating class
coffee6['rating_class'].value_counts()


# In[34]:



# Statistics of non-numeric variables

# Number of unique customers
print('\nNumber of unique customers : {}'.format(len(coffee6['reviewerID'].unique())))
      
# Number of unique products
print('\nNumber of unique products : {}'.format(len(coffee6['asin'].unique())))
      
# Review number per unique customer
print('\nReview per customer: {}'.format((len(coffee6)/len(coffee6['reviewerID'].unique()))))      

# Review number per unique product 
print('\nReview per product: {}'.format((len(coffee6)/len(coffee6['asin'].unique()))))


# In[35]:



# Read statistic summary of numeric variables
coffee6[['Rating']].describe()


# In[36]:


# PLOT NUMBER OF REVIEWS FOR TOP 20 BRANDS  
##########################################

brands = df["brand"].value_counts()
plt.figure(figsize=(12,8))
brands[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 20 Brands")
plt.xlabel('Brand Name')
plt.ylabel('Number of Reviews')


# In[37]:


## PLOT NUMBER OF REVIEWS FOR BOTTOM 20 BRANDS  
##########################################

brands = coffee6["brand"].value_counts()
# brands.count()
plt.figure(figsize=(12,8))
brands[-20:].plot(kind='bar')
plt.title("Number of Reviews for Bottom 20 Brands")
plt.xlabel('Brand Name')
plt.ylabel('Number of Reviews')


# In[38]:


##########################################
## PLOT NUMBER OF REVIEWS FOR TOP 20 PRODUCTS  
##########################################

products = coffee6["title"].value_counts()
plt.figure(figsize=(12,8))
products[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 50 Products")
plt.xlabel('Product Name')
plt.ylabel('Number of Reviews')


# In[39]:


coffee_fplot = coffee6.groupby(['month'])['Rating'].mean()
coffee_fplot


# In[40]:


# Histogram of Ratings
print(coffee6['Rating'].describe())
px.histogram(coffee6, x="Rating", nbins=30, title = 'Histogram of Ratings')


# In[41]:


# Total numbers of ratings in the home and kitchen product reviews
plt.figure(figsize = (10,6))
sns.countplot(coffee6['Rating'])
plt.title('Total Review Numbers for Each Rating', color='r')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()

# Customer totals for each rating class
coffee6['Rating'].value_counts()


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (10,6))

coffee6.groupby('Rating').Rating.count()
coffee6.groupby('Rating').Rating.count().plot(kind='pie',autopct='%1.1f%%',startangle=90,explode=(0,0.1,0,0,0),)


# In[43]:


#data=review_df.copy()
word_count=[]
for s1 in coffee6.review_text:
    word_count.append(len(str(s1).split()))


# In[44]:



plt.figure(figsize = (8,6))

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x="Rating",y=word_count,data=coffee6)
plt.xlabel('Rating')
plt.ylabel('Review Length')

plt.show()


# In[45]:


#Since there are outliers in the above boxplot we are not able to clearly visualize.So remove the outliers 
plt.figure(figsize = (8,6))

sns.boxplot(x="Rating",y=word_count,data=coffee6,showfliers=False)
plt.xlabel('Rating')
plt.ylabel('Review Length')

plt.show()


# In[46]:


# Total numbers of ratings in coffee product reviews
plt.figure(figsize = (8,6))
sns.countplot(coffee6['rating_class'])
plt.title('Total Review Numbers for Each Rating Class', color='r')
plt.xlabel('Rating Class')
plt.ylabel('Number of Reviews')
plt.show()

# Customer totals for each rating class
coffee6['rating_class'].value_counts()


# In[47]:


##################################################################
# Total review for every year in the Headphone product
#####################################################################
plt.figure(figsize = (12,8))
sns.countplot(coffee6['year'])
plt.title('Total Review Numbers for Each Year', color='r')
plt.xlabel('year')
plt.ylabel('Number of Reviews')
plt.show()

# Customer totals for each rating class
coffee6['year'].value_counts()


# In[48]:


# How many unique customers do we have in the dataset?
print('Number of unique customers: {}'.format(len(coffee6['reviewerID'].unique())))


# In[49]:


# Visualizations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS

# Datetime
from datetime import datetime


# In[50]:



# How many unique customers in each year?
unique_cust = coffee6.groupby('year')['reviewerID'].nunique()

# Plot unique customer numbers in each year
plt.figure(figsize = (10,6))
unique_cust.plot(kind='bar', rot = 0, color='m')
plt.title('Unique Customers in Each Year', color='k', size = 14)
plt.xlabel('Year')
plt.ylabel('Unique Customer Numbers')
plt.show()

# Print unique customer numbers in each year
print(unique_cust)


# In[51]:


# unique customers for each "rating class"
a = list(coffee6.groupby(['rating_class'])['reviewerID'].unique())  

# number of customers
a2 = [len(a[0]),len(a[1])] 

# number of reviews for each "rating class"
b = list(coffee6['rating_class'].value_counts())              

uniq_cust_rate = pd.DataFrame({'rating_class': ['bad', 'good'],
                               'number_of_customers': a2,
                               'number_of_reviews': sorted(b)})
print(uniq_cust_rate)


# In[52]:


# Print number of unique home and kitchen products in the dataset
print('Number of coffee products: {}'.format(len(coffee6['asin'].unique())))


# In[53]:


# How many unique products in each year?
unique_prod = coffee6.groupby('year')['asin'].nunique()

# Plot unique product numbers in each year
plt.figure(figsize = (10,6))
unique_prod.plot(kind='bar', rot =0, color = 'c')
plt.title('Unique Products in Each Year', color = 'k', size = 14)
plt.xlabel('Year')
plt.ylabel('Unique Product Numbers')
plt.show()

# Print unique product numbers in each year
print(unique_prod)


# In[54]:


##########################################
## DISTRIBUTION OF RATING IN PRODUCTS
########################################## 
plt.figure(figsize = (8,6))

coffee6_a = coffee6.copy()
coffee6_a = coffee6_a[np.isfinite(coffee6_a['Rating'])]
grp = coffee6_a.groupby('asin')
counts = grp.asin.count()        # number of reviews by each critic
means = grp.Rating.mean()     # average freshness for each critic

means[counts > 5].hist(bins=10, edgecolor='w', lw=1)
plt.xlabel("Average Rating per product")
plt.ylabel("Number of products")
plt.show()


# In[55]:


## Rating FOR LENGTH OF TEXT
########################################## 
plt.figure(figsize = (15,8))

coffee6_a['text_len'] = coffee6_a.review_text.apply(len)
maxTextLen = max(coffee6_a.text_len)
coffee6_a.groupby(pd.cut(coffee6_a['text_len'], np.arange(0,maxTextLen+1000,1000)))['Rating'].count().plot(kind='bar',color='b')
plt.xlabel("length of review text")
plt.ylabel("Class Rating")
plt.title("Relationship between 'Product Rating' and 'Length of review text'")
plt.ylim([0, 1])

plt.show()


# In[56]:



print("For all of the reviewers, they came from {} products.".format(coffee6.asin.nunique()))
print("\n")

top10_coffee = coffee6.groupby('asin').size().reset_index().sort_values(0, ascending = False).head(10)
top10_coffee.columns = ['Product', 'Counts']
print("Top 10 Products ranked by review counts")
top10_coffee


# In[57]:


top10_list = top10_coffee['Product'].tolist()
top10 = coffee6[coffee6.asin.isin(top10_list)]

fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
ax = sns.violinplot(x = 'asin', 
                    y = 'Rating', 
                    data = top10, 
                    order = top10_list,
                    linewidth = 2) 
plt.suptitle('Distribution of Ratings by Product') 
plt.xticks(rotation=90);


# In[58]:



print("For all of the reviewers, they came from {} brand products.".format(coffee6.brand.nunique()))
print("\n")

top10_coffee1 = coffee6.groupby('brand').size().reset_index().sort_values(0, ascending = False).head(10)
top10_coffee1.columns = ['Product Brand', 'Counts']
print("Top 10 Products Brand ranked by review counts")
top10_coffee1


# In[62]:


top10_list = top10_coffee1['Product Brand'].tolist()
top10 = coffee6[coffee6.brand.isin(top10_list)]

fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
ax = sns.violinplot(x = 'brand', 
                    y = 'Rating', 
                    data = top10, 
                    order = top10_list,
                    linewidth = 2) 
plt.suptitle('Distribution of Ratings by Product') 
plt.xticks(rotation=90);


# Text PreProcessing

# In[63]:


import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

# import libraries
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import re,random,os
import seaborn as sns
from nltk.corpus import stopwords
import string
from pprint import pprint as pprint

# spacy for basic processing, optional, can use nltk as well(lemmatisation etc.)
import spacy

#gensim for LDA
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# In[64]:


# tokenize using gensims simple_preprocess
def sent_to_words(sentences, deacc=True):  # deacc=True removes punctuations
    for sentence in sentences:
        yield(simple_preprocess(str(sentence)))

# conver to list
data=coffee6['review_text'].values.tolist()
data_words=list(sent_to_words(data))

#sample
print(data_words[3])


# In[65]:


len(str(data_words))


# In[66]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[67]:


#Remove Stopwords, Make Bigrams and Lemmatize

# NLTK Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'really', 're', 'would', 'use', 'go','be','ever','coffee'])


# In[68]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN','VERB','ADV','ADJ']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[69]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[70]:


import spacy

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN','VERB','ADV'])


# In[71]:


print(data_words_nostops[3])

print("Clean token after stopwords: ",len(str(data_words_nostops)))


# In[72]:


print(data_words_bigrams[3])

print("Clean token after bigrams: ",len(str(data_words_bigrams)))


# In[73]:


print(data_lemmatized[3])

print("Lemmatization-Noun,Verb,Adverb & Adjective): ",len(str(data_lemmatized)))


# In[74]:


import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])


# In[75]:


len(str(corpus))


# Text Visualization

# In[76]:


# function to plot most frequent terms
from nltk import FreqDist

def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()


# In[77]:


#Raw tokens: The most common words

freq_words(coffee6['review_text'])


# Base LDA Model

# In[78]:


# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)


# In[79]:


#the keywords for each topic and the weightage(importance) of each keyword using lda_model.print_topics()

from pprint import pprint

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[80]:


#Compute Model Perplexity and Coherence Score

from gensim.models import CoherenceModel

# Compute Coherence Score: the higher the better
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


# In[82]:


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=coffee):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row=row[0]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant highest weighted topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# In[83]:


df_sent_topic_keywords=format_topics_sentences(ldamodel=lda_model,corpus=corpus,texts=data_lemmatized)
df_dominant_topic=df_sent_topic_keywords.reset_index()
df_dominant_topic.columns=['DocumentNo','Dominant_Topic','Perc_Contribution','Topic_Keywords','texts']

df_dominant_topic.sample(10)


# In[84]:


print(df_dominant_topic.groupby('Dominant_Topic').count())


# In[85]:


# showing best relevant document under each topic
topic_sentences_df =pd.DataFrame()
df_topic_sents_grped=df_dominant_topic.groupby('Dominant_Topic')

for i,grp in df_topic_sents_grped:
    topic_sentences_df=pd.concat([topic_sentences_df,grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)
    
    
#reset index
topic_sentences_df.reset_index(drop=True,inplace=True)

#Format
topic_sentences_df.columns=['Document No','Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

topic_sentences_df.head()


# In[86]:


# Number of Documents for Each Topic
topic_counts = df_sent_topic_keywords['Dominant_Topic'].value_counts()
print(topic_counts)
# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)
print(topic_contribution)


# In[87]:


# Concatenate Column wise
df_dominant_topics = pd.concat([ topic_counts, topic_contribution], axis=1,)

# Show
df_dominant_topics.reset_index(inplace=True)

# Change Column names
df_dominant_topics.columns = ['Topic id', 'Num_Documents', 'Perc_Documents']

df_dominant_topics


# In[88]:


coffee7=pd.merge(df_dominant_topic,coffee6,left_index=True, right_index=True)

coffee7.head()


# In[89]:


coffee7.to_csv(r'C:\Users\Khadijah Jasni\Desktop\topic-lda.csv')


# In[90]:


topics = [[(term, round(wt, 3)) for term, wt in lda_model.show_topic(n, topn=20)] for n in range(0, lda_model.num_topics)]


# In[91]:


topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], columns = ['Term'+str(i) for i in range(1, 21)], index=['Topic '+str(t) for t in range(1, lda_model.num_topics+1)]).T
topics_df.head()


# In[92]:


# set column width
pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, lda_model.num_topics+1)] )
topics_df


# In[93]:



# import wordclouds
from wordcloud import WordCloud

# initiate wordcloud object
wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

# set the figure size
plt.rcParams['figure.figsize'] = [20, 15]

# Create subplots for each topic
for i in range(10):

    wc.generate(text=topics_df["Terms per Topic"][i])
    
    plt.subplot(5, 4, i+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(topics_df.index[i])

plt.show()


# In[95]:


def compute_coherence_values(dictionary,corpus,texts,start,limit,step):
    coherence_vals=[]
    model_list=[]
    
    for num_topics in range(start,limit,step):
        # building LDA Model
        model=gensim.models.LdaMulticore(corpus=corpus,id2word=dictionary,
                                              num_topics=num_topics,random_state=100,
                                              chunksize=100,passes=10,per_word_topics=True)

        model_list.append(model)
        
        coherencemodel=CoherenceModel(model=model,texts=texts,dictionary=dictionary,coherence='c_v')
        
        coherence_vals.append(coherencemodel.get_coherence())
    return model_list,coherence_vals

model_list,coherence_vals=compute_coherence_values(dictionary=id2word,
                                                   corpus=corpus,texts=data_lemmatized,
                                                   start=2,limit=20,step=4)


# In[96]:


import matplotlib.pyplot as plt

# visualize the optimal LDA Model
limit=20
start=2
step=4
x=range(start,limit,step)

plt.plot(x,coherence_vals)
plt.xlabel('Num_topics')
plt.ylabel('Coherence score')
plt.legend(('coh'),loc='best')
plt.show()


# In[97]:


for m, cv in zip(x,coherence_vals):
    print("num topics: ",m,'has coherence value of :',round(cv,4))


# In[101]:


# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)


# In[106]:


from matplotlib.ticker import FuncFormatter

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=240, sharey=True)

# Topic Distribution by Dominant Topics
ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
ax1.xaxis.set_major_formatter(tick_formatter)
ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
ax1.set_ylabel('Number of Documents')
ax1.set_ylim(0, 150000)

# Topic Distribution by Topic Weights
ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
ax2.xaxis.set_major_formatter(tick_formatter)
ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

plt.show()


# In[109]:


from gensim import corpora, models, similarities


lda_model.save('lda_model')


# SENTIMENT ANALYSIS

# In[111]:


coffee7.head()


# In[112]:


# case text as lowercase, remove punctuation, remove extra whitespace in string and on both sides of string

coffee7['remove_lower_punct'] = coffee7['review_text'].str.lower().str.replace("'", '').str.replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()

display(coffee7.head(10))


# In[114]:


get_ipython().system('pip install vaderSentiment')


# In[115]:


import collections
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# apply sentiment analysis
analyser = SentimentIntensityAnalyzer()

sentiment_score_list = []
sentiment_label_list = []

for i in coffee7['remove_lower_punct'].values.tolist():
    sentiment_score = analyser.polarity_scores(i)

    if sentiment_score['compound'] >= 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Positive')
    elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Neutral')
    elif sentiment_score['compound'] <= -0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Negative')
    
coffee7['sentiment'] = sentiment_label_list
coffee7['sentiment score'] = sentiment_score_list

display(coffee7.head(10))


# In[ ]:


coffee7.to_csv(r'C:\Users\Khadijah Jasni\Desktop\featurebased_sentiment.csv')


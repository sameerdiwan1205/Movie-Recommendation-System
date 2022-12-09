#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


#pip install openpyxl


# In[ ]:


#pip install -U scikit-learn


# In[ ]:


#pip install -U nltk


# In[ ]:


ls


# In[ ]:


data1 = pd.read_excel('converted data.xlsx')


# In[ ]:


data1.info()


# # Movie Recommendation Using Python

# In[ ]:


data = pd.read_excel('Top_10000_Movies.xlsx')


# In[ ]:


data


# In[ ]:


#Drop Unnecessary Columns
data = data.drop(['Unnamed: 13'], axis = 1)


# In[ ]:


data = data.drop(['Unnamed: 14'], axis = 1)


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


#Check Null Values
data.isnull().sum()


# In[ ]:


#Dropping Missing Values
data.dropna(inplace = True)


# In[ ]:


#Checking whether Values are Dropped or not
data.isnull().sum()


# In[ ]:


#Vectorizing TF-IDF 
TF_IDF = TfidfVectorizer(min_df = 5, max_df = 0.7 )
Vectorized_data = TF_IDF.fit_transform(data['overview'])


# In[ ]:


ndata = pd.DataFrame(Vectorized_data.toarray(), columns = TF_IDF.get_feature_names())


# In[ ]:


ndata.index = data['original_title']
ndata


# In[ ]:


#Finding Cosine Similarity
Cosine_Similarity = cosine_similarity(ndata)


# In[ ]:


Cosine_Similarity


# In[ ]:


#Cosine Similarity Dataframe
Cosine_df = pd.DataFrame(Cosine_Similarity, index = ndata.index, columns = ndata.index)


# In[ ]:


Cosine_df


# # Recommendation

# In[ ]:


Cosine_df.loc['Free Guy'].sort_values(ascending = False).head()


# # Demographic Filtering or Simple Recommendation system

# In[ ]:


data1


# In[ ]:


V = data1['vote_count'] # Number of vote
R = data1['vote_average'] #Rating
m = data1['vote_count'].quantile(0.8)
C = data1['vote_average'].mean()


# In[ ]:


C


# In[ ]:


m


# In[ ]:


ndata = data1.copy().loc[data1['vote_count'] >= m]
ndata.shape #filtering the movies for the chart


# In[ ]:


#Weigted average
data1['W_avarage'] = (V/(V+m) * R) + (m/(m+V) * C)


# In[ ]:


W_avarage


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()


# In[ ]:


#scaling Data With Popularity
Scaled_data = sc.fit_transform(data1[['popularity', 'W_avarage']])


# In[ ]:


W_data = pd.DataFrame(Scaled_data, columns = ['popularity', 'W_avarage'])
W_data.index = data1['original_title']


# In[ ]:


#taking 50% Weigthed averahe and 70% popularity
W_data['score'] = W_data['W_avarage']*0.5 + W_data['popularity'].astype('float64')*0.7


# In[ ]:


#Now sorting the data
W_data_sorted = W_data.sort_values(by = 'score', ascending = False)


# In[ ]:


#Recommendation
W_data_sorted.head(10)


# # Collborative Algorithm

# In[ ]:


ratings = pd.read_csv('rating.csv')
movies = pd.read_csv('movie.csv')


# In[ ]:


ratings.info()


# In[ ]:


movies.info()


# In[ ]:


ratings


# In[ ]:


ratings['userId'].nunique()#unique user id


# In[ ]:


ratings['movieId'].nunique()#unique movie id


# In[ ]:


ratings['rating'].nunique()#unique rating


# In[ ]:


movies.head()


# In[ ]:


ndata = pd.merge(ratings, movies, on = 'movieId')


# In[ ]:


ndata.head()


# In[ ]:


#taking movies only with 100 or more rating
a_rating = ndata.groupby('title').agg(mr = ('rating','mean'),nr = ('rating', 'count')).reset_index() #number of rating
rt_100 = a_rating[a_rating['nr']>100]
rt_100.info()


# In[ ]:


#only 8531 has more than 100 ratings


# In[ ]:


rt_100.sort_values(by='nr', ascending = False).head() #checking most popular movies


# In[ ]:


#now merging remaining movies with ratings
n_rt_100 = pd.merge(ndata,rt_100[['title']], on = 'title')
n_rt_100.info()


# In[ ]:


#creating matrix
matrix = n_rt_100.pivot_table(index= 'userId', columns = 'title', values = 'rating')


# In[ ]:


#data normalization
matrix_norm = matrix.subtract(matrix.mean(axis =1), axis = 'rows')
matrix_norm.head()


# In[ ]:


similar_mat = matrx_norm.T.corr() #similarity  matrix


# In[ ]:


#cosine similarity
user_cos_simi = cosine_similarity(matrix_norm.fillna(0)
user_cos_simi


# In[ ]:


#picking the user to find similar user

uid = 2
similar_mat.drop(index = uid, inplace = True)
user_cos_simi.head()


# In[ ]:


#number of similar user
n = 5

#threshhold similarity

tr = 0.4


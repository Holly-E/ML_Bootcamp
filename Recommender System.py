"""
Content based recommender system
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names=['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('Data/u.data', sep='\t', names=column_names) #tab separated data
movie_titles = pd.read_csv('Data/Movie_Id_Titles')
df = pd.merge(df, movie_titles, on='item_id')
sns.set_style('white')

#print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
#print(ratings.head())
#ratings['rating'].hist(bins=70)
#sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=.5)

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
#print(starwars_user_ratings.head())

# How similar to the movie matrix based on each user and their rating
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
#print(corr_starwars.head())

corr_starwars = corr_starwars.join(ratings['num of ratings'])
#print(corr_starwars.head())
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())

similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head())

plt.show()
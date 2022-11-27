import pandas as pd 
import numpy as np

def topMovies():
    path = './dataset/imdb_top_1000.csv'
    data = pd.read_csv(path)
    C = data['IMDB_Rating'].mean()
    m = data['No_of_Votes'].quantile(0.9)
    q_movies = data.copy().loc[data['No_of_Votes'] >= m]
    def weighted_rating(x, m=m, C=C):
        v = x['IMDB_Rating']
        R = x['No_of_Votes']
        return (v/(v+m) * R) + (m/(m+v) * C)
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    q_movies = q_movies.sort_values('score', ascending = False)
    # return q_movies[['Series_Title', 'IMDB_Rating']].head(8).to_json()
    # print(q_movies[['Series_Title']].head(8).to_json())
    return q_movies[['Series_Title']].head(8).to_json()
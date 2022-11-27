import os 
os.system('pip install scikit-learn')
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

def wantedMovie(title):
    tfidf = TfidfVectorizer(stop_words='english')
    path = './dataset/imdb_top_1000.csv'
    df = pd.read_csv(path)
    df['Overview'] = df['Overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['Overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['Series_Title']).drop_duplicates()
    def get_recommendations(title, cosine_sim=cosine_sim):
        
        idx = indices[title]

        sim_scores = list(enumerate(cosine_sim[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:8]

        movie_indices = [i[0] for i in sim_scores]

        return df['Series_Title'].iloc[movie_indices]
    return get_recommendations(title)
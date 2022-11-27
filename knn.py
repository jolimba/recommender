import os
import time

# data science imports
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import knn_methods as knn
from joblib import dump, load
# utils import
from fuzzywuzzy import fuzz

def collaborative(movie):
    return make_recommendation(
    model_knn=load('./predicts/model_knn.joblib'),
    data=load('./predicts/movie_user_mat_sparse.joblib'),
    fav_movie=movie,
    mapper=load('./predicts/movie_to_idx.joblib'),
    n_recommendations=6)


def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return 'Oops! No match is found'
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_recommendation(model_knn, data, fav_movie, mapper, n_recommendations=7):
    model_knn.fit(data)
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    reverse_mapper = {v: k for k, v in mapper.items()}
    response = []
    for i, (idx, dist) in enumerate(raw_recommends):
        response.append(reverse_mapper[idx])
    return response

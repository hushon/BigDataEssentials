'''
Hyounguk Shon
25-Oct-2019

Usage: python collaborativefiltering.py [file.txt]

Collaborative filtering algorithm.

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/ratings.txt
'''

import sys
import os
import numpy as np
import time
import itertools

def parse(l):
    '''
    Arg:
        l (str): a string of data.
    Return: 
        A list.
    '''
    userID, movieID, rating, timestamp = l.split(',')
    userID = int(userID)
    movieID = int(movieID)
    rating = float(rating)
    timestamp = int(timestamp)
    return [userID, movieID, rating, timestamp]

def cosine(x, y):
    '''
    Arg:
        x (np.ndarray): vector
        y (np.ndarray): vector
    Return:
        Cosine distance between x, y
    '''
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan
    else:
        x = np.nan_to_num(x).flatten()
        y = np.nan_to_num(y).flatten()
        return np.dot(x, y) / (norm(x) * norm(y))
    return result

def mean(x):
    '''
    Take mean of vector, but ignore np.nan values.
    '''
    x = np.array(x).flatten()
    result = np.nanmean(x)
    return result

def normalize_utility_matrix(M):
    '''
    Normalize matrix by subtracting each row with its mean.
    Arg:
        M (np.ndarray):
    Return:
        Row-wise normalized M
    '''
    return M - np.nanmean(M, axis=1)[..., np.newaxis]

def user_collaborative_filtering(M):

    '''normalize utility matrix'''
    normalized_M = normalize_utility_matrix(M)

    ''' user-based collaborative filtering '''
    # ID variables are denoted as index value
    U = 600-2
    top_ten_similar_users = sorted(range(normalized_M.shape[0]), key=lambda x: cosine(normalized_M[x, :], normalized_M[U, :]), reverse=True)[:10]

    # slim utility matrix by top-10 similar users
    N = M[top_ten_similar_users, :]
    
    # iterate though columns 0~999 of matrix N, and fill out M[U, 0:1000]
    prediction = M[U, 0:1000].copy()
    for j, movie_ratings in enumerate(N[:, 0:1000].T):
        assert np.isnan(prediction[j])
        prediction[j] = mean(movie_ratings)
    
    # print top-5 recommended movies among 1~1000
    top_five_recommended_movies = np.argsort(-prediction)[:5]
    for m in top_five_recommended_movies:
        print '{}\t{}'.format( m + 1, prediction[m])

def item_collaborative_filtering(M):
    
    ''' normalize utility matrix '''
    normalized_M = normalize_utility_matrix(M)

    ''' item-based collaborative filtering '''
    U = 600-2
    prediction = M[U, 0:1000].copy()

    ''' find top-10 similarly rated items 
    # X: target utility matrix (subject for prediction)
    # Y: source utility matrix (as a pre-condition) '''
    X = np.nan_to_num(normalized_M[:, 0:1000])
    Y = np.nan_to_num(normalized_M[:, 1000:])

    # calculate cosine similarity between source and target items
    Z = np.matmul(
        (X / (np.linalg.norm(X, axis=0) + 1e-6)).T,
        (Y / (np.linalg.norm(Y, axis=0) + 1e-6)))
    
    Q = np.argsort(-Z, axis=1)[:, :10]
    Q = Q + 1000

    # 10 similar movies of I => Q[i, :]

    for j in range(len(prediction)):
        prediction[j] = mean(M[U, Q[j, :]])

    # print top-five recommendtion using item-item collaborative filtering 
    top_five_recommended_movies = np.argsort(-prediction)[:5]
    for m in top_five_recommended_movies:
        print '{}\t{}'.format( m + 1, prediction[m])


def main():
    ''' parameters '''
    filepath = sys.argv[1]

    ''' read and parse dataset '''
    with open(filepath, 'r') as file:
        lines = file.readlines()

    ratings = map(parse, lines)

    ''' make utility matrix '''
    # initialize M with NaN value
    M = np.full(shape=(671, 164979), fill_value=np.nan, dtype=np.float)
    
    # iterate over dataset and fill out utility matrix
    for u, m, r, t in ratings:
        M[u-2, m-1] = r

    user_collaborative_filtering(M)

    item_collaborative_filtering(M)

if __name__ == '__main__':

    ''' sanity check '''
    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
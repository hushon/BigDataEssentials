'''
Hyounguk Shon
25-Oct-2019

Usage: python collaborativefiltering.py [file.txt]

Collaborative filtering algorithm.

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/ratings.txt
'''

import sys
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
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
    return userID, movieID, rating, timestamp

def par_map(f, x):
    with ThreadPoolExecutor() as executor:
        return executor.map(f, x)

def nandot(x, y):
    x = np.nan_to_num(x, copy=True)
    y = np.nan_to_num(y, copy=True)
    return x.dot(y)

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

def similarity_matrix(M, type='user'):
    '''
    Calculate cosine similarity for user(or item)-based filtering.
    '''
    M = np.nan_to_num(M, copy=True)
    if type == 'user':
        M = M / np.sqrt(np.sum(M**2, axis=1, keepdims=True))
        return M.dot(M.T)
    elif type == 'item':
        M = M / np.sqrt(np.sum(M**2, axis=0, keepdims=True))
        return M.T.dot(M)
    else:
        raise ValueError

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

def nanadd(x, y):
    '''
    Construct new matrix by completing values from x and y
    '''
    def nanadd_scalar(a, b):
        if np.isnan(a) and np.isnan(b):
            return np.nan
        else:
            return np.nan_to_num(a) + np.nan_to_num(b)
    return np.vectorize(nanadd_scalar)(x, y)

def nanmerge(master, branch):
    '''
    Construct new matrix by completing values from master and branch.
    Args:
        master (np.ndarray)
        branch (np.ndarray)
    '''
    def nanmerge_scalar(m, b):
        if not np.isnan(m):
            return m
        elif not np.isnan(b):
            return b
        else:
            return np.nan
    return np.vectorize(nanadd_scalar)(master, branch)

def user_collaborative_filtering(M, topk=10):
    '''
    user-based collaborative filtering

    Args:
        M (np.ndarray)
        topk (int): number of other users used for prediction
    Return:
        np.ndarray
    '''

    # normalize utility matrix
    normalized_M = normalize_utility_matrix(M)

    # calculate cosine similarity matrix to find similar users
    S = similarity_matrix(normalized_M, type='user') # TODO: what if entire row is NaN?
    S = S - np.diag(np.diag(S))
    top_ten_similar_users = np.argsort(-S, axis=1)[:, :topk]

    # make prediction by top-10 similar users
    prediction = np.nanmean(M.values[top_ten_similar_users, :], axis=1)
    prediction = pd.DataFrame(prediction, index=M.index, columns=M.columns)

    # fill utility matrix using prediction
    M[:] = nanmerge(M, prediction)

    result = []
    for i, r in M.loc[600, :].iteritems():
        if np.isnan(r) and 0<i<1000:
            result.append((i, prediction.loc[600, i]))

    tt = np.array(result, dtype=[('index', int), ('rating', float)])
    print np.sort(tt, order='rating')[:10]
    # print sorted(result, key=lambda (k, v): -v)[:5]

    # # iterate though columns 0~999 of matrix N, and fill out M[U, 0:1000]
    # prediction = M[U, 0:1000].copy()
    # for j, movie_ratings in enumerate(N[:, 0:1000].T):
    #     assert np.isnan(prediction[j])
    #     prediction[j] = mean(movie_ratings)

    # # print top-5 recommended movies among 1~1000
    # top_five_recommended_movies = np.argsort(-prediction)[:5]
    # for m in top_five_recommended_movies:
    #     print '{}\t{}'.format( m + 1, prediction[m])

def item_collaborative_filtering(M):
    pass

def main():
    ''' parameters '''
    filepath = sys.argv[1]

    ''' read and parse dataset '''
    with open(filepath, 'r') as file:
        lines = file.readlines()
        dataset = map(parse, lines)

    dataset = {(uid, mid): (r, t) for uid, mid, r, t in dataset}

    ''' make utility matrix '''
    rows = list(set([uid for (uid, _) in dataset.keys()]))
    cols = list(set([mid for (_, mid) in dataset.keys()]))

    # TODO: this can be parallelized
    matrix = pd.DataFrame(np.nan, index=rows, columns=cols, dtype=np.float)
    for (uid, mid), (r, t) in dataset.items():
        matrix.loc[uid, mid] = r

    user_collaborative_filtering(matrix)

if __name__ == '__main__':

    ''' sanity check '''
    assert os.path.exists(sys.argv[1]),  'Cannot find file.'

    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
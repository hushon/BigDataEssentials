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
import pandas as pd
import time

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

def similarity_matrix(M, mode='user'):
    '''
    Calculate cosine similarity for user(or item)-based filtering.
    Args:
        M (np.ndarray):
        mode (str): 'user' or 'item'
    Return:
        np.ndarray
    '''
    M = np.nan_to_num(M, copy=True)
    if mode == 'user':
        M = M / np.sqrt(np.sum(M**2, axis=1, keepdims=True))
        return M.dot(M.T)
    elif mode == 'item':
        M = M / np.sqrt(np.sum(M**2, axis=0, keepdims=True))
        return M.T.dot(M)
    else:
        raise ValueError

def normalize_utility_matrix(M, mode='user'):
    '''
    Normalize matrix by subtracting each row with its mean.
    Arg:
        M (np.ndarray):
        mode (str): 'user' or 'item'
    Return:
        Row-wise normalized M
    '''
    assert mode in ['user', 'item']
    if mode == 'user':
        return M - np.nanmean(M, axis=1)[..., np.newaxis]
    elif mode == 'item':
        return M - np.nanmean(M, axis=0)[np.newaxis, ...]
    else:
        raise ValueError

def nanmerge(master, branch):
    '''
    Merge two matrices by completing missing values of master matrix using branch matrix.
    If both entries have non-NaN value, prioritize the value in master matrix.
    Args:
        master (np.ndarray)
        branch (np.ndarray)
    Return:
        np.ndarray
    '''
    def nanmerge_scalar(m, b):
        if not np.isnan(m):
            return m
        elif not np.isnan(b):
            return b
        else:
            return np.nan
    return np.vectorize(nanmerge_scalar)(master, branch)

def collaborative_filtering(M, topk=10, mode='user'):
    '''
    Collaborative filtering to predict missing values in the matrix.
    Missing entries should be assigned by NaN.
    Mode decides between user-based or item-based filtering.
    Args:
        M (pd.DataFrame)
        topk (int): number of similar vectors used for prediction
        mode (str): 'user' or 'item'
    Return:
        pd.DataFrame
    '''
    assert mode in ['user', 'item']

    # normalize utility matrix
    normalized_M = normalize_utility_matrix(M, mode=mode)

    # calculate cosine similarity matrix to find similar users
    S = similarity_matrix(normalized_M, mode=mode) # TODO: what if entire row is NaN?
    np.fill_diagonal(S, 0.0)

    if mode == 'user':
        similar_users = np.argsort(-S, axis=1)[:, :topk]

        # make prediction by top-k similar users
        prediction = np.nanmean(M.values[similar_users, :], axis=1)
        prediction = pd.DataFrame(prediction, index=M.index, columns=M.columns)

    elif mode == 'item':
        similar_items = np.argsort(-S, axis=1)[:, :topk]

        # make prediction by top-k similar items
        prediction = np.nanmean(M.values[:, similar_items], axis=2)
        prediction = pd.DataFrame(prediction, index=M.index, columns=M.columns)

    # fill utility matrix using prediction
    M[:] = nanmerge(M, prediction)

    return M

def item_collaborative_filtering(M):
    pass

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    mode = 'user' # user- or item-based filtering
    query_uid = 600 # target user ID
    query_mid = slice(1, 999) # target movie ID
    topk = 10
    topn = 20

    ''' read and parse dataset '''
    with open(filepath, 'r') as file:
        lines = file.readlines()
        dataset = map(parse, lines)

    dataset = {(uid, mid): (r, t) for uid, mid, r, t in dataset}

    ''' make utility matrix '''
    rows = list(set([uid for (uid, _) in dataset.keys()]))
    cols = list(set([mid for (_, mid) in dataset.keys()]))

    # TODO: parallelize this operation
    matrix = pd.DataFrame(np.nan, index=rows, columns=cols, dtype=np.float)
    for (uid, mid), (r, t) in dataset.items():
        matrix.loc[uid, mid] = r

    '''try to predict missing values by collaborative filtering'''
    matrix = collaborative_filtering(matrix, topk, mode=mode)

    '''print recommendations'''
    pred = matrix.loc[query_uid, query_mid]
    for u, r in pred.sort_values(ascending=False)[:topn].iteritems():
        print '{}\t{}'.format(u, r)

if __name__ == '__main__':

    ''' sanity check '''
    assert os.path.exists(sys.argv[1]), 'Cannot find file.'

    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
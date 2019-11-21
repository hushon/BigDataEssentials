'''
Hyounguk Shon
25-Oct-2019

Usage: python collaborativefiltering.py [source_file.txt] [target_file.txt]

Collaborative filtering algorithm.

Example source file: http://www.di.kaist.ac.kr/~swhang/ee412/ratings.txt
Example target file: http://www.di.kaist.ac.kr/~swhang/ee412/ratings_test.txt
'''

import sys
import os
import numpy as np
import pandas as pd
import time
import itertools
from collections import Counter

def parse(l):
    '''
    Arg:
        l (str): a string of data.
    Return: 
        A list.
    '''
    user, item, rating, timestamp = l.split(',')
    user = int(user)
    item = int(item)
    rating = float(rating) if rating != '' else rating
    timestamp = int(timestamp)
    return [user, item, rating, timestamp]

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

def normalize_rows(M):
    '''
    Normalize matrix by subtracting each row with its mean.
    Arg:
        M (np.ndarray):
    Return:
        Row-wise normalized M
    '''
    assert isinstance(M, pd.DataFrame) and M.ndim == 2
    mean = np.nanmean(M, axis=1)
    stddev = np.nanstd(M, axis=1)
    return M.sub(mean, axis=0).divide(stddev, axis=0)

def user_collaborative_filtering(M, top_n):
    assert isinstance(M, pd.DataFrame) and M.ndim == 2

    '''normalize utility matrix'''
    normalized_M = normalize_rows(M)

    ''' user-based collaborative filtering '''
    # find top-10 similar users for each user
    X = np.nan_to_num(M)
    denom = np.linalg.norm(X, axis=1)[..., np.newaxis] + 1e-6

    # calculate cosine similarity between user pairs
    Z = X.dot(X.T).divide(denom, axis=0).divide(denom, axis=1)
    print Z

    Q = np.argsort(-Z, axis=1)[:, 1:11]
    assert Q.shape == (671, 10)
    top_ten_similar_users_list = list(Q)

    # fill in blanks utility matrix by user-based collaborative filtering
    N = M.copy()
    # predict using top-10 similar users
    K = np.nanmean(N[top_ten_similar_users_list, :], axis=1)
    mask = np.isnan(N)
    N = K*mask + np.nan_to_num(N) # this part isn't correct

    N = np.nan_to_num(N)
    return N


def main():
    ''' parameters '''
    filepath_trainset = sys.argv[1]
    filepath_testset = sys.argv[2]
    filepath_output = './output3.txt' # name of output file
    top_n = 10 # top-n similar user-user collaborative filtering
    userIDs = list(range(1, 672, 1)) # list of unique user ID
    itemIDs = list(range(1, 164980, 1)) # list of unique item ID

    ''' read and parse dataset '''
    with open(filepath_trainset, 'r') as file:
        lines = file.readlines()
        trainset = map(parse, lines)

    with open(filepath_testset, 'r') as file:
        lines = file.readlines()
        testset = map(parse, lines)

    ''' make utility matrix '''
    # initialize utility matrix M with NaNs
    M = pd.DataFrame(np.nan, index=userIDs, columns=itemIDs)
    assert M.shape == (671, 164979)
    
    # iterate over dataset and fill out utility matrix
    for u, i, r, t in trainset:
        M.ix[i, u] = r

    ''' do user-user collaborative filtering '''
    N = user_collaborative_filtering(M, top_n)

    ''' iterate through queries and write output to file '''
    with open(filepath_output, mode='wb') as file:
        for userID, movieID, _, timestamp in testset:
            prediction = N[userID - 2, movieID - 1]
            file.write('{},{},{},{}'.format(userID, movieID, prediction, timestamp))
            file.write('\n')
    

if __name__ == '__main__':

    ''' sanity check '''
    assert os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
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
    rating = float(rating) if rating != '' else rating
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
    return (M - np.nanmean(M, axis=1)[..., np.newaxis]) / np.nanstd(M, axis=1)[..., np.newaxis]

def user_collaborative_filtering(M):
    '''normalize utility matrix'''
    normalized_M = normalize_utility_matrix(M)

    ''' user-based collaborative filtering '''
    # find top-10 similar users for each user
    X = np.nan_to_num(M)
    denom = np.linalg.norm(X, axis=1)[..., np.newaxis]

    Z = np.matmul(
        X / (denom + 1e-6), 
        (X / (denom + 1e-6)).T
    )

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

def item_collaborative_filtering(M):
    
    ''' normalize utility matrix '''
    normalized_M = normalize_utility_matrix(M)

    ''' item-based collaborative filtering '''
    U = 600-2
    prediction = M[U, 0:1000].copy()

    X = np.nan_to_num(normalized_M[:, 0:1000])
    Y = np.nan_to_num(normalized_M[:, 1000:])

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
    filepath_trainset = sys.argv[1]
    filepath_testset = sys.argv[2]
    filepath_output = './output3.txt'

    ''' read and parse dataset '''
    with open(filepath_trainset, 'r') as file:
        lines = file.readlines()
        ratings = map(parse, lines)

    with open(filepath_testset, 'r') as file:
        lines = file.readlines()
        ratings_test = map(parse, lines)

    ''' make utility matrix '''
    # initialize M with NaN value
    M = np.full(shape=(671, 164979), fill_value=np.nan, dtype=np.float)
    
    # iterate over dataset and fill out utility matrix
    for u, m, r, t in ratings:
        M[u-2, m-1] = r

    ''' do user-user collaborative filtering '''
    N = user_collaborative_filtering(M)

    ''' iterate through queries and write output to file '''
    with open(filepath_output, mode='wb') as file:
        for userID, movieID, _, timestamp in ratings_test:
            prediction = N[userID - 2, movieID - 1]
            file.write('{},{},{},{}'.format(userID, movieID, prediction, timestamp))
            file.write('\n')
    

if __name__ == '__main__':

    ''' sanity check '''
    assert os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
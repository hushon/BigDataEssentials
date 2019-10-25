'''
Hyounguk Shon
25-Oct-2019

Usage: spark-submit collaborativefiltering.py [file.txt]

Collaborative filtering algorithm.

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/ratings.txt
'''

import sys
import os
from pyspark import SparkConf, SparkContext
import numpy as np
from numpy.linalg import norm
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
    x = np.nan_to_num(x).flatten()
    y = np.nan_to_num(y).flatten()
    assert len(x) == len(y)
    result = np.dot(x, y) / (norm(x) * norm(y))
    assert not np.isnan(result)
    return result

def mean(x):
    '''
    Take mean of vector, but ignore np.nan values.
    '''
    x = np.array(x).flatten()
    return np.mean([v for v in x if not np.isnan(v)])

def normalize_utility_matrix(M):
    '''
    Normalize matrix by subtracting each row with its mean.
    Arg:
        M (np.ndarray):
    Return:
        Normalized M
    '''
    means = []
    for i in range(M.shape[0]):
        means.append(mean(M[i, :]))
        M[i, :] = M[i, :] - means[i]
    assert len(means) == M.shape[0]
    return M, np.array(means)

def denormalize_utility_matrix(M, means):
    '''
    Denormalize matrix by adding each row with its mean.
    Arg:
        M (np.ndarray):
    Return:
        Denormalized M
    '''
    for i in range(len(M)):
        M[i, :] = M[i, :] + means[i]
    return M

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    nExecuter = 8
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    ''' read and parse dataset '''
    with open(filepath, 'r') as file:
        lines = file.readlines()

    ratings = map(parse, lines)
    
    # extract distinct userIDs and movieIDs
    ratings_rdd = sc.parallelize(ratings, nExecuter)
    user_list = ratings_rdd.map(lambda (u,m,r,t): u).distinct().collect()
    movie_list = ratings_rdd.map(lambda (u,m,r,t): m).distinct().collect()

    # make LUTs for mapping ID to index
    # user_LUT: userID -> i
    # movie_LUT: movieID -> j
    user_LUT = {id:idx for idx, id in enumerate(user_list)}
    user_LUT_transpose = {idx:id for idx, id in enumerate(user_list)}
    assert len(user_LUT) == len(user_list)
    movie_LUT = {id:idx for idx, id in enumerate(movie_list)}
    movie_LUT_transpose = {idx:id for idx, id in enumerate(movie_list)}
    assert len(movie_LUT) == len(movie_list)
    # userID: 2~672 -> 0~670
    # movieID: 1~164979 -> 0~164978

    ''' make utility matrix '''
    # initialize M with NaN value
    M = np.full(shape=(len(user_list), len(movie_list)), fill_value=np.nan, dtype=np.float)
    
    # iterate over dataset and fill out utility matrix
    for u, m, r, t in ratings:
        M[user_LUT[u], movie_LUT[m]] = r
    
    # normalize utility matrix
    M, means = normalize_utility_matrix(M)

    ''' top-10 similar users to U '''
    U = user_LUT[600]
    print U
    top_ten_similar_users = sorted(range(M.shape[0]), key=lambda x: cosine(M[x], M[U]), reverse=True)[:10]
    print top_ten_similar_users

    # slim utility matrix by top-10 similar users
    N = M[top_ten_similar_users, :]
    means_N = means[top_ten_similar_users]

    ''' user-based collaborative filtering '''
    # iterate though columns 0~999 of matrix N, and fill out M[U, 0:999]
    for j, movie_ratings in enumerate(N[:, [movie_LUT[j] for j in range(1, 1000, 1)]].T):
        assert np.isnan(M[U, j])
        M[U, j] = mean(movie_ratings)
    
    sorted(range(len(M[U, 0:999])), key=lambda j: M[U, j], reverse=True)


if __name__ == '__main__':

    ''' sanity check '''
    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
'''
Hyounguk Shon
07-Nov-2019

Usage: python pagerank.py [file.txt]

PageRank algorithm.

Example source file: http://www.di.kaist.ac.kr/~swhang/ee412/graph.txt
'''

import sys
import os
# import numpy as np
# import time
from pyspark import SparkConf, SparkContext


def parse(l):
    '''
    Arg:
        l (str)
    Return: 
        A tuple
    '''
    head, tail = l.split()
    head = int(head)
    tail = int(tail)
    return (head, tail)

def markov_matrix(M):
    '''
    divide each column vector by its sum
    '''
    len_i = len(M)
    len_j = len(M[0])

    for j in range(len_j):
        # accumulate column sum
        col_sum = 0.0
        for i in range(len_i):
            col_sum += M[i][j]
        # divide column by sum
        for i in range(len_i):
            M[i][j] = M[i][j]/col_sum
            print col_sum
    return M

def sca_vec_mul(a, v):
    '''
    scalar-vector multiplication
    '''
    result = [a*x for x in v]
    return result

def vec_vec_add(u, v):
    '''
    vector-vector addition
    '''
    assert len(u) == len(v)
    result = [x+y for x, y in zip(u, v)]
    return result

def mat_vec_mul(M, v, sc):
    '''
    Perform matrix-vector multiplication by map-reduce.
    Args:
        M (list): 2-dim array
        v (list): 1-dim vector
    Result:
        A list vector
    '''
    pairs = []
    for i in range(len(M)):
        for j in range(len(M[i])):
            pairs.append( (i, M[i][j]*v[j]) )
    pairs = sc.parallelize(pairs)
    pairs = pairs.reduceByKey(lambda v1, v2: v1 + v2)
    result = sorted(pairs.collect())
    result = map(lambda (k, v): v, result)
    return result

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    beta = 0.9
    n_nodes = 1000
    sc = SparkContext(conf=SparkConf())

    ''' read and parse dataset '''
    with open(filepath, 'r') as file:
        lines = file.readlines()
        dataset = map(parse, lines)

    # v = np.ones((n_nodes, 1), dtype=np.float) / n_nodes
    # e_n = np.ones((n_nodes, 1), dtype=np.float) / n_nodes
    # M = np.zeros((n_nodes, n_nodes), dtype=np.float)
    v = [1.0/n_nodes] * n_nodes
    e_n = [1.0/n_nodes] * n_nodes
    M = [[0.0]*n_nodes]*n_nodes

    '''generate transition matrix'''
    for (head, tail) in dataset:
        # M[tail-1, head-1] = 1.0
        M[tail-1][head-1] = 1.0
        # print tail, head, M[tail-1][head-1]

    # print np.sum(M, axis=0)
    return 0

    # M = M / np.sum(M, axis=0)[np.newaxis, ...]
    M = markov_matrix(M[:])
    # assert np.all(np.allclose(np.sum(M, axis=0), 1.0))

    ''' iteratively update pagerank '''
    for _ in range(50):
        # v = beta*np.matmul(M, v) + (1-beta)*e_n
        temp = mat_vec_mul(M, v, sc)
        a = sca_vec_mul(beta, temp)
        b = sca_vec_mul(1-beta, e_n)
        v = vec_vec_add(a, b)

if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    # starttime = time.time()
    main()
    # print 'Executed in: {}'.format(time.time()-starttime)
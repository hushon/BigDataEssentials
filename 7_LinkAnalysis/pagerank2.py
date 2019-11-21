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
    j, i = l.split()
    j = int(j)
    i = int(i)
    return ((i, j), 1.0)

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
    n_workers = 8
    sc = SparkContext(conf=SparkConf())

    ''' '''
    M_init = []
    for i in range(1, n_nodes+1, 1):
        for j in range(1, n_nodes+1, 1):
            M_init.append(((i, j), 0.0))

    ''' read and parse dataset '''
    lines = sc.textFile(filepath, n_workers).map(parse).distinct() # (i, j) -> 1.0
    M_init = sc.parallelize(M_init, n_workers)
    V = sc.parallelize([(j, 1.0/n_nodes) for j in range(1, n_nodes+1, 1)], n_workers)
    bE = sc.parallelize([(j, (1.0 - beta)/n_nodes) for j in range(1, n_nodes+1, 1)], n_workers)

    ''' initialize M and V '''
    M = M_init.union(lines).reduceByKey(lambda x, y: x+y)
    col_sum = M.map(lambda ((i, j), v): (j, v)).reduceByKey(lambda x, y: x+y).mapValues(lambda x: 1.0/x).flatMap(lambda (j, v): [((i, j), v) for i in range(1, n_nodes+1, 1)])
    M = M.union(col_sum).reduceByKey(lambda x, y: x*y)
    # print M.map(lambda ((i, j), v): (j, v)).reduceByKey(lambda x, y: x+y).sortByKey().collect()
    # return 0

    for _ in range(50):
        v_enlarge = V.flatMap(lambda (j, v): [((i, j), v) for i in range(1, n_nodes+1, 1)])
        MV = M.union(v_enlarge).reduceByKey(lambda x, y: x*y).map(lambda ((i, j), v): (i, v)).reduceByKey(lambda x, y: x+y)
        bMV = MV.mapValues(lambda v: beta*v)
        V = bMV.union(bE).reduceByKey(lambda x, y: x+y)

    print V.map(lambda (i, v): v).reduce(lambda x, y: x+y)
    return 0

if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    # starttime = time.time()
    main()
    # print 'Executed in: {}'.format(time.time()-starttime)
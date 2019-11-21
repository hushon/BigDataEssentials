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
import time
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

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    beta = 0.9
    n_nodes = 1000
    n_workers = None
    sc = SparkContext(conf=SparkConf())

    ''' read and parse dataset '''
    M = sc.textFile(filepath, n_workers).map(parse).distinct() # (i, j) -> 1.0
    V = sc.parallelize([(j, 1.0/n_nodes) for j in range(1, n_nodes+1, 1)], n_workers)
    bE = sc.parallelize([(j, (1.0 - beta)/n_nodes) for j in range(1, n_nodes+1, 1)], n_workers)

    ''' initialize M and V '''
    col_sum = M.map(lambda ((i, j), v): (j, v)).reduceByKey(lambda x, y: x+y)
    M = M.map(lambda ((i, j), v): (j, (i, v))).join(col_sum).map(lambda (j, ((i, v1), v2)): ((i, j), v1/v2))

    ''' iterate V '''
    for _ in range(10):
        bMV = M.map(lambda ((i, j), v): (j, (i, v))).join(V).map(lambda (j, ((i, v1), v2)): (i, beta*v1*v2)).reduceByKey(lambda x, y: x+y)
        V = bMV.union(bE).reduceByKey(lambda x, y: x+y)

    ''' print result '''
    print V.map(lambda (i, v): v).reduce(lambda x, y: x+y)
    print len(V.sortByKey().collect())

if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
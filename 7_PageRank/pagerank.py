'''
Hyounguk Shon
07-Nov-2019

Usage: spark-submit pagerank.py [file.txt]

PageRank algorithm.

Example dataset: http://www.di.kaist.ac.kr/~swhang/ee412/graph.txt
'''

import sys
import os
import time
import pyspark


def parse(l):
    '''
    Arg:
        l (str): string of <source node> <target node>
    Return: 
        A tuple
    '''
    j, i = l.split()
    j = int(j)
    i = int(i)
    return (i, j)

def sparseMatVecProduct(M, V):
    '''
    Matrix-vector product between a sparse matrix and a sparse vector.
    Arg:
        M (pyspark.rdd.RDD): a sparse matrix; (i, j) -> entry
        V (pyspark.rdd.RDD): a sparse vector; j -> entry
    Return:
        A sparse vector; i -> entry
    '''
    assert isinstance(M, pyspark.rdd.RDD) and isinstance(V, pyspark.rdd.RDD)
    return M.map(lambda ((i, j), v): (j, (i, v))).join(V)\
        .map(lambda (j, ((i, v1), v2)): (i, v1*v2))\
        .reduceByKey(lambda x, y: x+y)

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    beta = 0.9 # taxation parameter
    max_iter = 10
    n_workers = None
    topN = 10
    sc = pyspark.SparkContext(conf=pyspark.SparkConf())

    ''' read and parse dataset '''
    M = sc.textFile(filepath, n_workers).map(parse)
    node_ids = M.flatMap(lambda (i, j): [i, j]).distinct().collect()

    ''' initialize M and V '''
    M = M.distinct().map(lambda (i, j): ((i, j), 1.0))
    col_sum = M.map(lambda ((i, j), v): (j, v)).reduceByKey(lambda x, y: x+y)
    M = M.map(lambda ((i, j), v): (j, (i, v))).join(col_sum).map(lambda (j, ((i, v1), v2)): ((i, j), v1/v2))
    V = sc.parallelize([(j, 1.0/len(node_ids)) for j in node_ids], n_workers)
    E = sc.parallelize([(j, 1.0/len(node_ids)) for j in node_ids], n_workers)

    ''' iterate V '''
    for _ in range(max_iter):
        MV = sparseMatVecProduct(M, V)
        V = MV.mapValues(lambda v: beta*v)\
            .union(E.mapValues(lambda v: (1.0 - beta)*v))\
            .reduceByKey(lambda x, y: x+y)

    ''' print result '''
    for (i, v) in V.takeOrdered(topN, key=lambda (i, v): -v):
        print '{}\t{:.5f}'.format(i, v)

if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
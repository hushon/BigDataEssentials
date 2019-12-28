'''
Hyounguk Shon
21-Sep-2019

Usage: spark-submit commonfriends.py [file.txt]

Social network data mining for friend recommendation.
Find pairs of potential friends based on number of mutual friends.

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/hw1q1.zip
'''

import sys
from pyspark import SparkConf, SparkContext
import os
import time

def parse(l):
    '''
    Arg:
        l (str)
    Return:
        A tuple, (vertex, neighbors)
    Example:
        input: '9   0,6085,18972,19269'
        output: (9, [0, 6085, 18972, 19269])
    '''
    vertex, neighbors = l.split('\t')
    vertex = int(vertex)
    if len(neighbors)>0:
        neighbors = neighbors.split(',')
        neighbors = map(int, neighbors)
    else:
        neighbors = []
    assert '' not in neighbors
    return (vertex, neighbors)

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    topN = 10
    n_workers = None
    sc = SparkContext(conf=SparkConf())

    ''' read file and parse graph '''
    lines = sc.textFile(filepath, n_workers)
    graph = lines.map(parse).filter(lambda (k, v): v)

    ''' calculate number of mutual friends between each pair, filter those that are immediate friends '''
    temp = graph.flatMapValues(lambda v: v)
    potentialfriend = temp.join(temp).filter(lambda (k, v): v[0] < v[1]).map(lambda (k, v): (v, 1))
    immediatefriend = graph.flatMap(lambda (k, v): [(tuple(sorted((k, x))), None) for x in v])
    potentialfriend = potentialfriend.union(immediatefriend).groupByKey().filter(lambda (k, v): None not in v).mapValues(sum)

    ''' print top-N potential friends '''
    for (pf1, pf2), v in potentialfriend.takeOrdered(topN, key=lambda (k, v): -v):
        print '{}\t{}\t{}'.format(pf1, pf2, v)

if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]),  'Cannot find file.'

    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
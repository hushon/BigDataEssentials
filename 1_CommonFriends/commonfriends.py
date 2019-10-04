'''
Hyounguk Shon
21-Sep-2019

Usage: spark-submit commonfriends.py [file.txt]

Find potential friends pair based on number of mutual friends.

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/hw1q1.zip
'''

import re
import sys
from pyspark import SparkConf, SparkContext
import os
import itertools
import time

def split_me_and_friends(l):
    '''
    Arg:
        l (str)
    Return: 
        A tuple of user and friends.
    Example:
        Input: '9   0,6085,18972,19269'
        Output: (9, [0, 6085, 18972, 19269])
    '''
    me, friends = l.split('\t')
    me = int(me)
    if len(friends)>0:
        friends = friends.split(',')
        friends = map(int, friends)
    else: 
        friends = []
    assert '' not in friends
    return (me, friends)

def generate_candidates(line):
    '''
    Arg:
        line (tuple): Friend relation
    Example:
        line: (9, [0, 6085, 18972, 19269])
        dist0: [((0,9), -9999), ((9,6085), -9999), ...]
        dist1: [((0,6085), 1), ((0,18972), 1),...]
    '''
    inf = 9999
    me, friends = line
    
    ''' dist0: pair of distance zero. One is directly friend of another.
    dist1: pair of distance one. Each share at least one common friend. '''
    dist0 = [(tuple(sorted((me, friend))), -inf) for friend in friends]
    dist1 = [(pair, 1) for pair in itertools.permutations(friends, 2) if pair[0] <= pair[1]]
    return dist0 + dist1

def main():
    nExecutor = 8
    topN = 10

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    ''' parse and generate graph '''
    lines = sc.textFile(sys.argv[1], nExecutor)
    graph = lines.map(split_me_and_friends)

    ''' generate candidates tuples '''
    candidates = graph.flatMap(generate_candidates)

    ''' find potential friends '''
    candidates = candidates.reduceByKey(lambda x, y: x + y)
    potentialfriends = candidates.filter(lambda x: x[1] > 0)
    potentialfriends = potentialfriends.sortBy(lambda x: x[1], ascending=False)

    ''' print top-N potential friends '''
    for k, v in potentialfriends.collect()[:topN]:
        print '{}\t{}\t{}'.format(k[0], k[1], v)

if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    # print 'Executed in: {}'.format(time.time()-starttime)
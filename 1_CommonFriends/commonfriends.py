'''
Hyounguk Shon
21-Sep-2019

Usage: spark-submit commonfriends.py [file.txt]

Social network data mining for friend recommendation.
Find pairs of potential friends based on number of mutual friends.

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
        A tuple of user and friends. (me, [friends])
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

def generate_mutualfriends(line):
    '''
    Arg:
        line (tuple): Friend relation
    Example:
        line: (9, [0, 6085, 18972, 19269])
        dist1: [((0,9), [None]), ((9,6085), [None]), ...]
        dist2: [((0,6085), [9]), ((0,18972), [9]),...]
    '''
    me, friends = line
    
    ''' dist1: pair of distance one. One is an immediate friend of another.
    dist2: pair of distance two. Each share at least one common friend. '''
    dist1 = [(tuple(sorted((me, friend))), [flag]) for friend in friends]
    dist2 = [(pair, [me]) for pair in itertools.permutations(friends, 2) if pair[0] < pair[1]]
    return dist1 + dist2

def main():
    ''' parameters '''
    global flag
    flag = None
    nExecutor = 8
    topN = 10

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    ''' parse and generate graph '''
    lines = sc.textFile(sys.argv[1], nExecutor)
    graph = lines.map(split_me_and_friends)

    ''' generate mutual friends '''
    mutualfriends = graph.flatMap(generate_mutualfriends)
    mutualfriends = mutualfriends.reduceByKey(lambda x, y: x + y)

    ''' filter mutual friends if one is an immediate friend of another '''
    potentialfriends = mutualfriends.filter(lambda (k, v): flag not in v)

    ''' sort by number of mutual friends '''
    potentialfriends = potentialfriends.map(lambda (k, v): (k, len(v)))
    potentialfriends = potentialfriends.sortBy(lambda (k, v): v, ascending=False)

    ''' print top-N potential friends '''
    for (pf1, pf2), v in potentialfriends.collect()[:topN]:
        print '{}\t{}\t{}'.format(pf1, pf2, v)

if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
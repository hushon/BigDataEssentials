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

def connecteds_and_commons(line):
    '''
    friends: [((me, friend1), -9999999999), ...]
    friendsoffriend: [((friend1, friend2), 1), ...]
    '''
    me, friends = line
    friends = [((me, friend), -9999999999) for friend in friends]
    friendsoffriend = [(pair, 1) for pair in itertools.permutations(friends, 2)]
    return friends + friendsoffriend

def main():

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    lines = sc.textFile(sys.argv[1])
    lines = lines.map(split_me_and_friends)
    lines = lines.flatMap(connecteds_and_commons)
    lines = lines.reduceByKey(lambda x, y: x + y)
    lines = lines.filter(lambda l: l[1] > 0)
    lines = lines.sortByKey()

    for pair, count in lines.collect():
        print pair, count

if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]),  'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
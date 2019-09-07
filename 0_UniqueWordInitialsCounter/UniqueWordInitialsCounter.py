'''
Hyounguk Shon
08-Sep-2019

Usage: spark-submit UniqueWordInitialsCounter.py [file.txt]

Search for unique words within given text file, 
and print number of occurences that has each alphabet initial.

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/pg100.txt
'''

import re
import sys
from pyspark import SparkConf, SparkContext
import os

def main():
    
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    lines = sc.textFile(sys.argv[1])
    words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))
    words = words.map(lambda x: x.lower())
    words = words.distinct()

    initials = words.map(lambda x: x[0:1])
    initials = initials.filter(lambda x: bool(x) and x in 'abcdefghijklmnopqrstuvwxyz')

    pairs = initials.map(lambda i: (i, 1))
    counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
    counts = counts.sortByKey()

    for key, value in counts.collect():
        print '%c   %d' % (key, value)

if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]), '[!] Cannot find %s' % sys.argv[1]

    main()
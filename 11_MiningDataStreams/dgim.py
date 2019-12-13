'''
Hyounguk Shon
11-Dec-2019

Usage: python dgim.py [stream.txt] k0 k1 ... kn

Implementation of DGIM algorithm.

Example: http://www.di.kaist.ac.kr/~swhang/ee412/stream.txt
'''

import sys
import os
import time

def parse(l):
    '''
    Arg:
        l (str)
    Return:
        An integer.
    '''
    l = int(l)
    return l

class bucket:
    def __init__(self, right, size):
        self.end_timestamp = right
        self.ones_count = size

    def __add__(self, other):
        '''merge self with other'''
        self.end_timestamp = max(self.end_timestamp, other.end_timestamp)
        self.ones_count += other.ones_count
        return self

    def __repr__(self):
        return "<DGIM bucket: right at {}, size {}>".format(\
            self.ones_count, \
            self.end_timestamp)

class dgim_queue:
    '''
    Queue-like data structure that self-merges buckets to satisfy invariant.
    '''
    def __init__(self):
        self.buckets = [[]]

    def push(self, b):
        '''
        Push a bucket into queue and merge if there exists more than two
        buckets of the same size.
        Args:
            b (bucket): new bucket to push into queue
        '''
        self.buckets[0].insert(0, b)
        
        # merge buckets
        for i in range(len(self.buckets)):
            if len(self.buckets[i]) > 2:
                x = self.buckets[i].pop()
                y = self.buckets[i].pop()
                try:
                    self.buckets[i+1].insert(0, x + y)
                except IndexError:
                    self.buckets.append([])
                    self.buckets[i+1].insert(0, x + y)

    def query(self, t):
        '''count occurrence of one within recent bits down until timestamp t
        Args:
            t (int): earliest timestamp to trace
        '''
        ones_count = 0
        for q in self.buckets:
            for b in q:
                if b.end_timestamp < t:
                    break
                else:
                    ones_count += b.ones_count
                    last_count = b.ones_count
        ones_count -= last_count / 2.0
        return ones_count

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    queries = map(int, sys.argv[2:])

    ''' sanity check '''
    assert os.path.exists(sys.argv[1]), 'Cannot find file.'

    ''' read and parse dataset '''
    with open(filepath, 'r') as file:
        window = file.read().splitlines()
        window = map(parse, window)

    ''' generate buckets '''
    queue = dgim_queue()
    for t, x in enumerate(window):
        if x:
            queue.push(bucket(t, 1))

    ''' query and print results '''
    for k in queries:
        result = queue.query(len(window) - k)
        print "{}".format(result)

if __name__ == '__main__':
    starttime = time.time()
    main()
    # print 'Executed in: {}'.format(time.time()-starttime)

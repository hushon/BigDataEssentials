'''
Hyounguk Shon
11-Dec-2019

Usage: python dgim.py [training.csv] [testing.csv]

Implementation of DGIM algorithm.

Example: http://www.di.kaist.ac.kr/~swhang/ee412/stream.txt
'''

import sys
import os
import time
import numpy as np

def parse(l):
    '''
    Arg:
        l (str)
    Return: 
        An integer.
    '''
    l = int(l)
    return l

class block:
    def __init__(self, end, size):
        self.end_timestamp = end 
        self.ones_count = size

    def __add__(self, other):
        '''merge self with other'''
        self.end_timestamp = max(self.end_timestamp, other.end_timestamp)
        self.ones_count += other.ones_count
        return self

    def __repr__(self):
        return "size-{} block ending at {}".format(self.ones_count, self.end_timestamp)

class auto_merging_block_queue:
    '''
    Queue-like data structure that self-merges blocks to satisfy DGIM invariant.
    '''
    def __init__(self):
        self.blocks = [[]]

    def push(self, b):
        self.blocks[0].insert(0, b)
        
        for i in range(len(self.blocks)):
            if len(self.blocks[i]) > 2:
                x = self.blocks[i].pop()
                y = self.blocks[i].pop()
                try:
                    self.blocks[i+1].insert(0, x + y)
                except:
                    self.blocks.append([])
                    self.blocks[i+1].insert(0, x + y)

    def query(self, t):
        '''count number of ones in recent bits until timestamp t'''
        ones_count = 0
        for q in self.blocks:
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

    ''' generate blocks '''
    queue = auto_merging_block_queue()

    for t, x in enumerate(window):
        if x:
            queue.push(block(t, 1))

    ''' query '''
    for k in queries:
        t_last = len(window) - 1
        print "{}".format(queue.query(t_last - k + 1))

if __name__ == '__main__':
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
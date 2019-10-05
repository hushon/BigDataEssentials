'''
Hyounguk Shon
01-Oct-2019

Market-basket analysis using A-priori algorithm.
Find frequent pairs from baskets of items.

Usage: python apriori.py [file.txt]

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/browsing.txt
'''

import re
import sys
import os
import itertools
import time

class triangularMatrix:
    '''
    Implements Triangular Matrix data structure.
    Triangular matrix is indexed by [i, j], while j>i. 
    '''
    def __init__(self, n, fillvalue=0):
        size = int(n*(n-1)/2.0)
        self.n = n
        self.matrix = [fillvalue]*size

    def _check_indices(self, key):
        i, j = key
        if i<0 or j<0 or not j>i: 
            raise IndexError('index out of range: {}'.format([i, j]))
    
    def _linear_index(self, key):
        n = self.n
        i, j = key
        return int(i*(n-(i+1)/2.0)) + j - i - 1
    
    def __getitem__(self, key):
        n = self.n
        self._check_indices(key)
        index = self._linear_index(key)
        return self.matrix[index]

    def __setitem__(self, key, item):
        n = self.n
        self._check_indices(key)
        index = self._linear_index(key)
        self.matrix[index] = item

    def __len__(self):
        return len(self.matrix)

    def aslist(self):
        return self.matrix
    
    def index_aslist(self):
        n = self.n
        trimatrix_index = triangularMatrix(n)
        for i in range(n-1):
            for j in range(i+1, n, 1):
                trimatrix_index[i, j] = (i, j)
        return trimatrix_index.aslist()

def main():

    ''' parameters '''
    support_thres = 200
    topK = 10

    '''read and preprocess data '''
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    baskets = map(lambda l: re.split(r'[^\w]+', l), lines)
    baskets = map(lambda x: filter(None, x), baskets)

    ''' Pass 1: identify frequent items '''
    
    # LUT1: item ID string -> index 0 ~ n-1
    LUT1 = {}
    LUT1_transpose = {}
    index = 0
    for basket in baskets:
        for item in basket:
            if item not in LUT1:
                LUT1.update({item: index})
                LUT1_transpose.update({index: item})
                index += 1
    
    # count frequency of items
    counts = [0]*len(LUT1)
    for basket in baskets:
        for item in basket:
            counts[LUT1[item]] += 1

    # filter by frequent singletons
    # LUT2: frequent item ID string -> index 0 ~ m-1
    LUT2 = {}
    LUT2_transpose = {}
    index = 0
    for i, count in enumerate(counts):
        if count >= support_thres:
            item = LUT1_transpose[i]
            LUT2.update({item: index})
            LUT2_transpose.update({index: item})
            index += 1

    nFrequentItems = len(LUT2)
    print nFrequentItems


    ''' Pass 2: count pairs of frequent items '''
    tri_matrix = triangularMatrix(nFrequentItems, fillvalue=0)
    for basket in baskets:
        basket = filter(lambda x: x in LUT2, basket) # filter by frequent items
        basket = map(lambda x: LUT2[x], basket)
        for i, j in itertools.permutations(basket, 2):
            if i < j:
                tri_matrix[i, j] += 1

    nFrequentPairs = sum(k>=support_thres for k in tri_matrix.aslist())
    print nFrequentPairs


    ''' print top-K frequent pairs '''

    # index of sorted triangular matrix
    descendingIndex = sorted(tri_matrix.index_aslist(), reverse=True, key=lambda (i, j): tri_matrix[i, j])

    # translate index value to item ID string
    for triindex in descendingIndex[:topK]:
        i, j = triindex
        item_i = LUT2_transpose[i]
        item_j = LUT2_transpose[j]

        print '{}\t{}\t{}'.format(item_i, item_j, tri_matrix[triindex])


if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]), 'Cannot find file.'

    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
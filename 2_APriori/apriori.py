'''
Hyounguk Shon
21-Sep-2019

Finding frequent pairs using A-priori algorithm.

Usage: python apriori.py [file.txt]

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/browsing.txt
'''

import re
import sys
import os

def main():

    ''' parameters '''
    support_thres = 200

    '''read and preprocess data '''
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    baskets = map(lambda l: re.split(r'[^\w]+', l), lines)
    baskets = map(lambda x: filter(None, x), baskets) # remove empty string

    ''' Pass 1 '''
    LUT1 = {} # id string -> 1 ~ n
    LUT1_transpose = {} # 1 ~ n -> id string
    count = [] # integer -> frequency
    for basket in baskets:
        for item in basket:
            if item not in LUT1: 
                # update lookup table
                key, value = item, len(LUT1) + 1
                LUT1.update({
                    key: value
                    })
                
                LUT1_transpose.update({
                    value: key
                    })

                count.append(1) # update count
            else:
                # update count
                count[LUT1[item]-1] += 1

    ''' find singletons with support >= s '''
    LUT2 = {} # id string -> 1 ~ m
    LUT2_transpose = {} # 1 ~ m -> id string

    for item in LUT1:
        if count[LUT1[item]-1] >= support_thres:
            # update LUT2
            key, value = item, len(LUT2) + 1
            LUT2.update({
                key: value
                })

            LUT2_transpose.update({
                value: key
                })


    ''' Pass 2 '''
    m = len(LUT2)
    tri_matrix = [0] * int(m*(m-1)/2.0) # initialize matrix
    for basket in baskets:
        
        basket = filter(lambda x: x in LUT2, basket) # filter by frequent items
        basket = map(lambda x: LUT2[x], basket)

        for i in basket:
            for j in basket:
                if j > i:
                    tri_index = int((i-1)*(m-i/2.0) + j - i - 1)
                    tri_matrix[tri_index] += 1

    ''' print # of frequent items '''
    print m

    ''' print # of frequent pairs '''
    nFrequentPairs = 0
    for i in tri_matrix:
        if i >= support_thres: nFrequentPairs += 1
    print nFrequentPairs # number of frequent pairs

    ''' print top-10 frequent pairs '''

    # index of sorted triangular matrix
    descendingIndex = sorted(range(len(tri_matrix)), reverse=True, key=lambda i: tri_matrix[i])
    
    # make lookup table for converting index to (i, j)
    key2ij = []
    for i in range(m):
        for j in range(m):
            if j > i:
                key2ij.append((i, j))
    assert len(key2ij) == int(m*(m-1)/2.0)

    # convert (i, j) to item id string
    for index in descendingIndex[:10]:
        i, j = key2ij[index]

        item_i = LUT2_transpose[i]
        item_j = LUT2_transpose[j]
        tri_index = int((i-1)*(m-i/2.0) + j - i - 1)

        print '{}\t{}\t{}'.format(item_i, item_j, tri_matrix[index])


if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]), 'Cannot find file.'

    main()

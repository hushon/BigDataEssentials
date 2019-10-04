import re
import sys
import os
from itertools import combinations

def main():

    ''' parameters '''
    support_thres = 200

    '''read and preprocess data '''
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    baskets = map(lambda l: re.split(r'[^\w]+', l), lines)
    baskets = map(lambda x: filter(None, x), baskets) # remove empty string

    # find frequency of each item
    itemcount = {}
    for basket in baskets:
        for item in basket:
            if item not in itemcount:
                itemcount.update({item: 1})
            else:
                itemcount[item] += 1

    # identify frequent items
    frequentitems = []
    for item, count in itemcount.items():
        if count >= support_thres:
            frequentitems.append(item)

    # find frequency of each pair
    paircount = {}
    for basket in baskets:
        for pair in combinations(basket, 2):
            if pair not in paircount:
                paircount.update({pair: 1})
            else:
                paircount[pair] += 1

    # print top-10 frequent pairs


    print count



if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]), 'Cannot find file.'

    main()

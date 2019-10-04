'''
Hyounguk Shon
01-Oct-2019

Finding similar documents using MinHash and Locality-sensitive Hashing Algorithm

Usage: python lsh.py [file.txt]

Example text file: http://www.di.kaist.ac.kr/~swhang/ee412/articles.txt
'''

import re
import sys
import os
import numpy as np
import time

def get_shingles(string, k=3):
    '''
    Return k-shingles of the given string.
    Args: 
        string (str): input string
        k (int): length of shingles
    Return:
        A list containing all length-k shingles
    '''
    assert isinstance(k, int) and k>0

    return [string[i:i+k] for i in range(len(string)-k+1)]

def string_permutations(k, alphabet='abcdefghijklmnopqrstuvwxyz '):
    '''
    Return all possible permutation of length-k string, 
    allowing repeated alphabets.
    Args:
        k (int): length of a string
        alphabet (str): alphabets
    Return:
        A list containing every permutation of length-k string.
        e.g., ['aaa', 'aab', 'aac', ..., '  z', '   ']
    '''
    assert isinstance(k, int) and k>0

    # recursively call string_permutations
    if k == 1:
        return [i for i in alphabet]
    else:
        permutations = []
        for permutation in string_permutations(k-1, alphabet=alphabet):
            for i in alphabet:
                permutations.append(permutation + i)
        return permutations
    
def random_hashfunc(n):
    '''
    Return random hash function.

    Args:
        n (int): roughly number of buckets
    Returns:
        A random hash function whose number of bucket 
        is a prime number greater or equals to n.
    '''
    assert isinstance(n, int) and n>0

    c = get_prime_number(n)
    a = np.random.randint(1, c-1)
    b = np.random.randint(0, c-1)
    return lambda x: (a*x + b) % c

def get_prime_number(n):
    '''
    Return the smallest prime number larger than or equals to n.
    '''
    assert isinstance(n, int) and n>0

    p = n
    while True:
        # Check if p is a prime number
        r = [p%i for i in range(2, int(np.floor(np.sqrt(p)))+1, 1)]
        if 0 in r:
            p += 1
            continue
        else:
            break
    return p

def onehot(i, n):
    '''
    Transform an integer index to one-hot vector.
    Args:
        i (int): index ranging from 0 to n-1
        n (int): length of one-hot vector
    Returns:
        A length-n ndarray of 0's and 1.
    '''
    assert isinstance(i, int) and isinstance(n, int) and n>i>=0

    result = [0]*n
    result[i] = 1
    return result

def minhash(M, nHashfunc=120):
    '''
    Transform given matrix into minhash signature matrix.
    Assume shape of M is (nDocuments, nBinaryVector)
    Args:
        M (np.ndarray)
        nHashfunc (int): number of hash functions to apply
    Return:
        A np.ndarray with shape corresponding to (nCols, nHashfunc)
    '''
    assert isinstance(nHashfunc, int) and nHashfunc>0

    M = np.array(M)
    nCols, nRows = M.shape

    result = np.full((nCols, nHashfunc), fill_value=np.inf)
    hashfuncs = [random_hashfunc(nRows) for _ in range(nHashfunc)]

    for r, row in enumerate(M.transpose()):
        h = [hashfunc(r) for hashfunc in hashfuncs] # hashed row index
        for c in np.where(row == 1):
            result[c] = np.minimum(result[c], h) # update result[c] by taking min signature
    result = result.astype(np.int)

    assert result.shape == (nCols, nHashfunc)
    return result

def LSH(M, b=6, r=20):
    '''
    Take a Min-hashed matrix M,
    and return column indices that turns out to be similar pairs.
    Args:
        b (int): number of bands
        r (int): number or rows per band
    Return:
        A list containing dictionaries corresponding to each band.
        Each dictionary contains buckets of similar pairs.
    '''
    M = np.array(M)
    nCols, nRows = M.shape
    assert isinstance(b, int), isinstance(r, int) and b*r == nRows

    result = []
    for i in range(0, nRows, r): # for each band
        band = M[:, i:i+r]

        # create hash map whose key is column vectors in each band,
        # and its value corresponds to column index of the vector
        band_dict = {}
        for colIdx, colVec in enumerate(band):
            colVec = tuple(colVec)
            if colVec not in band_dict:
                band_dict.update({colVec: [colIdx]})
            else:
                band_dict[colVec].append(colIdx)

        # append column indices of similar items
        for colIdxs in band_dict.values():
            if len(colIdxs) > 1:
                result.append(colIdxs)

    return result

def main():

    ''' read and preprocess data '''
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    lines = map(lambda l: l.split(' ', 1), lines) # split into article id and content
    ids = map(lambda l: l[0], lines) # article IDs
    lines = map(lambda l: l[1], lines) # contents

    lines = map(lambda l: re.compile(r'[^a-zA-Z ]').sub('', l), lines) # remove non-alphabetic characters
    lines = map(lambda l: l.lower(), lines) # transform string into lowercase


    ''' Transform documents into binary matrix representing shingles '''
    shingles = map(lambda l: get_shingles(l, k=3), lines) # [['Tro', 'roo', 'oop', 'ops', ...], ...]

    # LUT: DocID string -> column index
    LUT = {DocID:i for i, DocID in enumerate(ids)}
    LUT_transpose = {i:DocID for i, DocID in enumerate(ids)}

    shingles = map(lambda l: l, shingles) # [['Tro', 'roo', 'oop', 'ops', ...], ...]

    permutations = string_permutations(k=3)

    for i in range(len(shingles)):
        vector = map(lambda s: permutations.index(s), shingles[i]) # [[13, 29, 12, 30, ...], ...]
        result = [0]*len(permutations)
        for j in vector:
            result[j] = 1
        shingles[i] = result

    shingles = np.array(shingles)


    ''' generate minhash signature matrix '''
    M = minhash(shingles)

    ''' generate LSH matrix '''
    similar_pairs = LSH(M)

    for pair in similar_pairs:
        pair = [LUT_transpose[i] for i in pair]
        print '{}\t{}'.format(*pair)


if __name__ == '__main__':

    assert os.path.exists(sys.argv[1]), 'Cannot find file.'

    starttime = time.time()
    main()
    # print 'Executed in: {}'.format(time.time()-starttime)
'''
Hyounguk Shon
12-Nov-2019

Usage: python girvan-newman.py [file.csv]

Girvan-Newman algorithm.

Example source file: http://www.di.kaist.ac.kr/~swhang/ee412/paper_authors.csv
'''

import sys
import os
# import time
from pyspark import SparkConf, SparkContext

def parse(l):
    '''
    Arg:
        l (str)
    Return: 
        A tuple
    '''
    _, paper_id, author_id = l.split(',')
    paper_id = int(paper_id)
    author_id = int(author_id)
    return (paper_id, author_id)

# def cartesian(x, y):
#     '''
#     Arg:
#         x (list)
#         y (list)
#     Return:
#         Cartesian product of x and y
#     '''
#     assert len(x)>0 and len(y)>0
#     result = []
#     for i in x:
#         for j in y:
#             result.append((i, j))
#     return result

class Node:
    def __init__(self, name):
        self._name = name
        self._in_nodes = set()
        self._out_nodes = set()
        self._visited = False

    def __repr__(self):
        return str(self._name)

    @property
    def in_nodes(self):
        return self._in_nodes

    @property
    def out_nodes(self):
        return self._out_nodes

    def append_in(self, n):
        self._in_nodes.add(n)

    def append_out(self, n):
        self._out_nodes.add(n)

    def set_visited(self):
        self._visited = True

    def is_visited(self):
        return self._visited


class Graph:
    def __init__(self):
        self.vertices_map = dict()

    def __repr__(self):
        return str(self.vertices_map)

    def add_vertice(self, key):
        self.vertices_map.update({key: Node(key)})

    def add_edge(self, key_out, key_in):
        vertice_out = self.vertices_map[key_out]
        vertice_in = self.vertices_map[key_in]
        vertice_out.append_out(vertice_in)
        vertice_in.append_in(vertice_out)
        assert self.vertices_map[key_in]._in is vertice_out

    def lookup_by_key(self, key):
        ''' key -> Node '''
        return return self.vertices_map[key]

    def vertices(self):
        return self.vertices_map.values()

def BFS(graph, root_key):
    '''
    transform graph into a DAG with root as given key
    Args:
        graph (Graph)
        root_key (int)
    Return:
        a graph, directed acyclic
    '''
    assert isinstance(Graph, graph)
    visited_nodes = set()
    curr_node = graph.lookup_by_key(root_key)
    queue = []

    while True:
        # push current node into set of visited nodes
        visited_nodes.add(curr_node)

        for n in curr_node.out_nodes:
            queue.insert(0, n)
    


    while 

def main():
    ''' parameters '''
    filepath = sys.argv[1]
    sc = SparkContext(conf=SparkConf())
    n_workers = 8

    ''' read and parse dataset '''
    lines = sc.textFile(filepath, n_workers)

    # remove csv column header
    header = lines.first()
    lines = lines.filter(lambda l: l != header)

    # parse
    paper2author = lines.map(parse) # (paper, author)

    ''' preprocess dataset '''
    # paper2authors = paper2author.groupByKey() # (paper, [authors])
    # paper2coauthors = paper2authors.flatMap(lambda (k, v): cartesian(v, v)).filter(lambda (k, v): k != v).distinct()
    # paper2authors = paper2authors.flatMap(lambda (k, v): [(k, i) for i in v])
    paper2authors = paper2author.join(paper2author)
    author2author = paper2authors.map(lambda (k, v): v)
    author2author = author2author.filter(lambda (k, v): k != v).distinct()
    author2coauthors = author2author.groupByKey() # (author, [coauthors])
    graph_LUT = author2coauthors.mapValues(list).collectAsMap() # {author: [coauthors]}

    ''' build graph from lookup table '''
    graph = Graph()

    for key in graph_LUT.keys():
        graph.add_vertice(key)

    for key_out, value in graph_LUT.iteritems():
        for key_in in value:
            graph.add_edge(key_out, key_in)
            graph.add_edge(key_in, key_out)

    ''' Girvan-Newman algorithm for calculating betweeness of edges '''
    visited_nodes = set()

    root = sc.parallelize(graph_LUT.keys(), n_workers)
    level_1_node = root.flatMap(lambda n: graph_LUT[n]).distinct()
    level_2_node = level_1_node.flatMap(lambda n: graph_LUT[n]).distinct()

    vertices = sc.parallelize(graph.vertices, n_workers)
    bfs = vertices.map(vertex => (vertex, BFS(vertex, graph)))
    betweenness = bfs.map(vertex => calculateBetweenness(vertex, graph))
    betweenness = betweenness.reduceByKey(lambda x, y: x+y).mapByValue(lambda x: x/2.0)






if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]), 'Cannot find file.'
    
    # starttime = time.time()
    main()
    # print 'Executed in: {}'.format(time.time()-starttime)
'''
Hyounguk Shon
12-Nov-2019

Usage: spark-submit girvan-newman.py [file.csv]

Girvan-Newman algorithm.

Example source file: http://www.di.kaist.ac.kr/~swhang/ee412/paper_authors.csv
'''

import sys
import os
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

def bfs(graph_LUT, root):
    ''' given an graph adjacency LUT and a root node, compute level of each node. '''
    visited = set([root])
    queue = [root]
    level = {root: 0}

    while len(queue) > 0:
        curr_node = queue.pop()
        for child_node in filter(lambda n: n not in visited, graph_LUT[curr_node]):
            queue.insert(0, child_node)
            level.update({child_node: level[curr_node] + 1})
            visited.add(child_node)
    return level

def betweenness(root, level_LUT, graph_LUT):
    '''(root, level_LUT, graph_LUT) -> [((edge1, edge2), betweenness), ...]'''
    
    ''' compute node weight '''
    # compute node weight
    node_weight = {k: 0 for k in graph_LUT.keys()}
    node_weight[root] = 1

    visited = []
    queue = [root]

    while len(queue) > 0:
        curr_node = queue.pop()
        if curr_node not in visited:
            for child_node in filter(lambda n: level_LUT[n] - level_LUT[curr_node] == 1, graph_LUT[curr_node]):
                queue.insert(0, child_node)
                node_weight[child_node] += node_weight[curr_node]
            visited.append(curr_node)

    ''' compute edge weight '''
    # visit node using visited stack
    edge_weight = []
    node_inflow = {k: 0.0 for k in graph_LUT.keys()}

    while len(visited) > 0:
        curr_node = visited.pop()

        child_weight_sum = sum([node_weight[n] for n in graph_LUT[curr_node] if level_LUT[curr_node] - level_LUT[n] == 1])

        for child_node in filter(lambda n: level_LUT[curr_node] - level_LUT[n] == 1, graph_LUT[curr_node]):
            weight = (1.0 + node_inflow[curr_node]) * node_weight[child_node] / child_weight_sum
            edge_weight.append((tuple(sorted((curr_node, child_node))), weight))
            node_inflow[child_node] += weight

    return edge_weight


def main():
    ''' parameters '''
    filepath = sys.argv[1]
    sc = SparkContext(conf=SparkConf())
    n_workers = None
    topN = 10

    ''' read and parse dataset '''
    lines = sc.textFile(filepath, n_workers)
    header = lines.first()
    lines = lines.filter(lambda l: l != header)
    paper2author = lines.map(parse) # paper -> author

    ''' preprocess dataset '''
    paper2authors = paper2author.join(paper2author)
    author2author = paper2authors.map(lambda (k, v): v)
    author2author = author2author.filter(lambda (k, v): k != v).distinct()
    author2coauthors = author2author.groupByKey()
    graph_LUT = author2coauthors.mapValues(list).collectAsMap() # author -> [coauthors]

    ''' bfs on graph '''
    roots = sc.parallelize(graph_LUT.keys(), n_workers)
    level_LUTs = roots.map(lambda root: (root, bfs(graph_LUT, root)) )

    ''' calculate betweenness of each edge '''
    btwns = level_LUTs.flatMap(lambda (root, level_LUT): betweenness(root, level_LUT, graph_LUT)).reduceByKey(lambda x, y: x+y).mapValues(lambda x: x/2.0)

    ''' print result '''
    for (id1, id2), v in btwns.takeOrdered(topN, key=lambda (k, v): -v):
        print '%d\t%d\t%.5f' % (id1, id2, v)

if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]), 'Cannot find file.'
    
    main()

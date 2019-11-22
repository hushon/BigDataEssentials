'''
Hyounguk Shon
12-Nov-2019

Usage: python girvan-newman.py [file.csv]

Girvan-Newman algorithm.

Example source file: http://www.di.kaist.ac.kr/~swhang/ee412/paper_authors.csv
'''

import sys
import os
import time
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
    ''' given an graph adjacency LUT and a root node, compute level of each node and DAG. '''
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
    # (root, level_LUT, graph_LUT) -> [((edge1, edge2), betweenness), ...]
    
    ''' compute node weight '''
    # make DAG using level
    DAG_LUT = dict()
    for k, v in graph_LUT.iteritems():
        try:
            DAG_LUT.update({k: filter(lambda n: level_LUT[n] - level_LUT[k] == 1, v)})
        except Exception as e:
            print k, v
            raise ValueError
        else:
            pass
        finally:
            pass

    # compute node weight
    node_weight = {k: 0 for k in DAG_LUT.keys()}
    node_weight[root] = 1

    visited = []
    queue = [root]

    while len(queue) > 0:
        curr_node = queue.pop()
        if curr_node not in visited:
            for child_node in DAG_LUT[curr_node]:
                queue.insert(0, child_node)
                node_weight[child_node] += node_weight[curr_node]
            visited.append(curr_node)

    ''' compute edge weight '''
    # make reverse directed DAG
    DAG_reverse_LUT = dict()
    assert len(level_LUT) == len(graph_LUT)
    for k, v in graph_LUT.iteritems():
        DAG_reverse_LUT.update({k: filter(lambda n: level_LUT[k] - level_LUT[n] == 1, v)})

    # visit node using visited stack
    edge_weight = []
    node_inflow = {k: 0.0 for k in graph_LUT.keys()}

    while len(visited) > 0:
        curr_node = visited.pop()

        child_weight_sum = sum([node_weight[n] for n in DAG_reverse_LUT[curr_node]])

        for child_node in DAG_reverse_LUT[curr_node]:
            weight = (1.0 + node_inflow[curr_node]) * node_weight[child_node] / child_weight_sum
            edge_weight.append((tuple(sorted((curr_node, child_node))), weight))
            node_inflow[child_node] += weight

    return edge_weight


def main():
    ''' parameters '''
    filepath = sys.argv[1]
    sc = SparkContext(conf=SparkConf())
    n_workers = None

    ''' read and parse dataset '''
    lines = sc.textFile(filepath, n_workers)

    # remove csv column header
    header = lines.first()
    lines = lines.filter(lambda l: l != header)

    # parse
    paper2author = lines.map(parse) # (paper, author)

    ''' preprocess dataset '''
    paper2authors = paper2author.join(paper2author)
    author2author = paper2authors.map(lambda (k, v): v)
    author2author = author2author.filter(lambda (k, v): k != v).distinct()
    author2coauthors = author2author.groupByKey() # (author, [coauthors])
    graph_LUT = author2coauthors.mapValues(list).collectAsMap() # {author: [coauthors]}

    # a,b,c,d,e,f,g,h,i,j,k = range(1,12,1)
    # graph_LUT = {a:[b,c,d,e], b:[a,c,f], c:[a,b,f], d:[a,g,h], e:[a,h], f:[b,c,i], g:[d,i,j], h:[e,d,j], i:[f,g,k], j:[h,g,k], k:[i,j]}
    level_LUT = bfs(graph_LUT, 1)
    betweenness(1, level_LUT, graph_LUT)
    return 0

    ''' bfs on graph '''
    roots = sc.parallelize(graph_LUT.keys(), n_workers)
    level_LUTs = roots.map(lambda root: (root, bfs(graph_LUT, root)) )

    ''' calculate betweenness of each edge '''
    btwns = level_LUTs.flatMap(lambda (root, level_LUT): betweenness(root, level_LUT, graph_LUT))#.reduceByKey(lambda x, y: x+y).mapValues(lambda x: x/2.0)

    print btwns.collect()



    # depth_1_nodes = roots.map(lambda : )

    # ''' build graph from lookup table '''
    # graph = Graph()

    # for key in graph_LUT.keys():
    #     graph.add_vertice(key)

    # for key_out, value in graph_LUT.iteritems():
    #     for key_in in value:
    #         graph.add_edge(key_out, key_in)
    #         graph.add_edge(key_in, key_out)

    # ''' Girvan-Newman algorithm for calculating betweeness of edges '''
    # visited_nodes = set()

    # root = sc.parallelize(graph_LUT.keys(), n_workers)
    # level_1_node = root.flatMap(lambda n: graph_LUT[n]).distinct()
    # level_2_node = level_1_node.flatMap(lambda n: graph_LUT[n]).distinct()

    # vertices = sc.parallelize(graph.vertices, n_workers)
    # bfs = vertices.map(vertex => (vertex, BFS(vertex, graph)))
    # betweenness = bfs.map(vertex => calculateBetweenness(vertex, graph))
    # betweenness = betweenness.reduceByKey(lambda x, y: x+y).mapByValue(lambda x: x/2.0)




if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]), 'Cannot find file.'
    
    starttime = time.time()
    main()
    print 'Executed in: {}'.format(time.time()-starttime)
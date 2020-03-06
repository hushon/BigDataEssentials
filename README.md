# Big Data Analysis with Spark

## Introduction

This repo demonstrates common big data analytics algorithms in python. The examples refer to the problems in Stanford's CS246 course.  
Some codes require Apache Spark API to leverage MapReduce style of workload parallelism.  

## Requirements

- Python 2.7
- Apache Spark 3.6
- NumPy
- Pandas

## Algorithms

| No | Description | Spark |
|---|---|---|
| 1 | Friend recommendation by mining social-network graphs | ✔️ |
| 2 | A-priori Algorithm: mining baskets for frequent itemsets |  |
| 3 | Locality-sensitive Hashing: finding similar items |  |
| 4 | K-means Clustering | ✔️ |
| 5 | Dimensionality Reduction: principal component analysis, CUR decomposition |  |
| 6 | Collaborative Filtering: mining ratings database for movie recommendation |  |
| 7 | PageRank | ✔️ |
| 8 | Girvan-Newman Algorithm: community detection in social-network graphs | ✔️ |
| 9 | Support Vector Machine |  |
| 10 | Deep Learning |  |
| 11 | DGIM algorithm: mining continuous stream of data |  |

## TODO

- [ ] Upgrade to Python 3
- [x] Pandas support

## References

- [Jure Leskovec, Anand Rajaraman, Jeff Ullman, *Mining of Massive Datasets*](http://www.mmds.org/)
- [Jure Leskovec, Michele Catasta, *CS246: Mining Massive Datasets*, Stanford](http://web.stanford.edu/class/cs246/)

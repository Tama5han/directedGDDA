# directedGDDA.py

This script is a collection of functions related to the directed GDDA.



## Usage

This script contains the following functions:

- `agreement(G1, G2)` : Compute the GDDA of two directed graphs G1 and G2.
- `graphlet_degree_distribution(G)` : Compute the GDD of directed graph G.
- `graphlet_decomposition(G)` : Compute the number of directed graphlets contained in directed graph G.
- `graphlet_degree(G)` : Compute graphlet degrees of all nodes in directed graph G.


### agreement

```python
import networkx as nx
import directedGDDA as dgdda


G1 = nx.DiGraph()
G1.add_edges_from([(0,1), (0,2), (1,2), (2,0), (2,3), (4,0), (4,3)])

G2 = nx.DiGraph()
G2.add_edges_from([(0,1), (1,2), (2,3), (3,0), (3,1), (3,4), (5,2)])


agreement = dgdda.agreement(G1, G2)
# 0.7715952439954236
```


### graphlet_degree_distribution

`gdd[i]` is the GDD of the i-th orbit.

```python
gdd = dgdda.graphlet_degree_distribution(G1)
# [array([1, 1, 3]),
#  array([1, 1, 3]),
#  array([2, 2, 1]),
#  array([3, 0, 2]),
#  array([2, 2, 1]),
     :
     :
```


### graphlet_decomposition

`decomposition[i]` is the number of the i-th directed graphlets contained in graph G1.

```python
decomposition = dgdda.graphlet_decomposition(G1)
# array([7, 4, 2, 2, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
#        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```


### graphlet_degree

`degrees[v][i]` is the degree of the i-th orbit for node v.

```python
degrees = dgdda.graphlet_degree(G1)
# {0: array([2, 2, 1, 2, 0, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
#         1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#  1: array([1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
#         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                        :
                        :
```

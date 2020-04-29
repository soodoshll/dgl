This is an example of distributed sampling using dgl.

### Design and pipeline

 1. Partition (1-halo)
 2. (?) Reorder nodes (in order to do global2local id conversion)
    
    - How about store a global2local mapping on every machine. Thus this step can be skipped.
    - I prefer this way. Since we can store the partition book on every client, why don't we save the global2local id mapping on every server?

### Question
 
 1. How to do layerwise sampling? 
 1. How to store the partition book efficiently?
 1. int32 and int64

 1. Euler has simple conversion. Edge id conversion takes a long time.

 ### Profile 

 Total time 30s

 1. sampling takes 11s.
 2. edge relabling takes 6s (It is redundant?)

``` 
g2l: 0.0663744 | sample: 10.7772 | l2g: 6.45502 (stage1: 0.538309, stage2:6.44553) | serialization: 0.493192
```

``` 
g2l: 0.0662261 | sample: 11.3448 | l2g: 0.713474 (stage1: 0.707863, stage2:0.708022) | serialization: 0.648562
```

```
t_split: 0.194995 | t_wait: 15.6748 | t_union: 2.05495
```

12 clients, without edge l2g coversion:

```
23568 KETPS
```

which is close to DGL under the single machine setting.

Euler
```
30271 KETPS
```


**No edge id l2g here**

ppi dataset

+ 4servers 12clients: 717059 ETPS
+ 4servers 1clients: 716944 ETPS


reddit dataset
+ 4servers 12clients: 1014 KETPS
+ 4servers 1clients: 1770 KEPTS
+ 1servers 1clients: 3718 KEPTS
+ 1servers 12clients: 21262 KEPTS
  - waiting: 19.94s union: 1.81s

+ 4servers 12clients ompthread=12 48686KEPTS
+ 12servers 12clients ompthread=4  47978KEPTS
+ 12servers 12clients ompthread=2  50773KEPTS
+ 12servers 12clients ompthread=1  50755KEPTS
+ 12servers 12clients ompthread=12 2230KEPTS

+ 48servers 12clients ompthread=1 35164KETPS

### Usage

 - `partition.py` is for random or metis partitioning. 
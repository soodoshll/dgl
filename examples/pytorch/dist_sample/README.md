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

### Usage

 - `partition.py` is for random or metis partitioning. 

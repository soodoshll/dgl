# save the ppi dataset as a dgl file 

import dgl
from dgl.data.utils import save_graphs

import os

data_dir = 'ppi'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

ppi_dataset = dgl.data.PPIDataset('train')
ppi_g = ppi_dataset.graph

file_name = 'ppi_train.dgl'
save_graphs(os.path.join(data_dir, file_name), [ppi_g])
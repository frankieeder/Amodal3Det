import numpy as np
import os

root = './layer_wise_outs0/'

get_layer_data = lambda f: np.load(root + f)
layer_outs = {}
for f in os.listdir(root):
    layer_outs[f[:-4]] = get_layer_data(f)
pass
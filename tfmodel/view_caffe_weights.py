import numpy as np

root = './layer_wise_outs0/'

get_layer_data = lambda f: np.load(root + f + '.npy')
conv1_2 = get_layer_data("conv1_2")
pool1 = get_layer_data("pool1")
conv2_1 = get_layer_data("conv2_1")
conv2_2 = get_layer_data("conv2_2")
pool2 = get_layer_data("pool2")
conv3_1 = get_layer_data("conv3_1")
x=2
from main import *
import numpy as np
import os

def get_caffe_layer_outputs_from_file(pkls_dir):
    layer_outs = {}
    for f in os.listdir(pkls_dir):
        layer_outs[f[:-4]] = np.load(os.path.join(pkls_dir, f))
    return layer_outs

root = './layer_wise_outs0/'
#layer_outs = get_caffe_layer_outputs_from_file(root)

tf_model = test_tf_implementation(
    cache_file="roidb_test_19_smol.pkl",
    weights_path="../pretrained/output/rgbd_det_iter_40000.h5",
    all_layer_outs=True
)

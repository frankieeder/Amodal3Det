from main import *
import numpy as np
import os
import pandas as pd

def load_npys_as_dict(pkls_dir):
    dict = {}
    for f in os.listdir(pkls_dir):
        dict[f[:-4]] = np.load(os.path.join(pkls_dir, f))
    return dict

def collapse_pickled_results(pickles_dir):
    results = []
    files = os.listdir(pickles_dir)
    for i in range(len(files)):
        file = pickles_dir + f"ROI_{i}.pkl"
        with open(file, 'rb') as fid:
            results.append(pickle.load(fid))
    with open('outs0.pkl', 'wb') as fid:
        pickle.dump(results, fid)

generate_results = False
if generate_results:
    tf_model = test_tf_implementation(
        cache_file="roidb_test_19_smol.pkl",
        weights_path="../pretrained/output/rgbd_det_iter_40000.h5",
        all_layer_outs=True
    )

#collapse_pickled_results("./tf_outs0/")
caffe_outs = load_npys_as_dict('./output/caffe_layer_wise_outs0/')
tf_outs = load_npys_as_dict('./output/tf_layer_wise_outs0/')
comparison_results = []
for caffe_name, caffe_out in caffe_outs.items():
    layer_prefix = caffe_name[:4]
    tf_name = caffe_name
    if layer_prefix == "conv" and int(caffe_name[4]) >= 3:  # Change to account for different names between corresponding layers
        tf_name = "relu" + tf_name[4:]
    if tf_name in tf_outs.keys():
        tf_out = tf_outs[tf_name]
        num_rois = len(tf_out)
        caffe_out = caffe_out[:num_rois]
        comparison_result = {}
        comparison_result['caffe_name'] = caffe_name
        comparison_result['num_rois'] = num_rois
        if layer_prefix == 'conv' or (layer_prefix == 'pool' and int(caffe_name[4]) <= 4): # Flip axes and trim length if pre-roi
            tf_out = np.moveaxis(tf_out, 3, 1)  # Switch axis order to match caffe
            tf_out = tf_out[:1]  # Only one image needed, all 5 layers should be identical
        try:
            sse = np.sum((caffe_out - tf_out) ** 2)
            comparison_result['sse'] = sse
            comparison_result['num_elements'] = caffe_out.size
            comparison_results.append(comparison_result)
        except:
            print(f"Failed layer: {caffe_name}")
            pass

comparison_results_df = pd.DataFrame(comparison_results)
comparison_results_df.to_csv("output/comparison_results0.csv")
pass





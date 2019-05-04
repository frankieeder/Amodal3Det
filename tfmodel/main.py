from model import *
from weights import *
from make_test_data import *
import os.path as osp
import pickle


SAVE_MODEL = True
tf_model = make_deng_tf_stucture()
#tf_model = load_weights(tf_model)
if SAVE_MODEL:
    out_dir = "deng_tensorflow.h5"
    tf_model.save(out_dir)

cache_file = ""
if osp.exists(cache_file):
    with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
    print('data is loaded from {}'.format(cache_file))
len(roidb)

test_net(tf_model, roidb)






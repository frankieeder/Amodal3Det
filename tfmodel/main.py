from model import *
from weights import *
from test import *
import os.path as osp
import pickle
from keras.models import load_model

SAVE_MODEL = False
LOAD_MODEL = False # TODO: Get this to work
model_dir = "deng_tensorflow.h5"
cache_file = "roidb_test_19_smol.pkl"
weights_path = "../pretrained/output/rgbd_det_iter_40000.h5"


# Define Model Structure
if LOAD_MODEL:
    tf_model = load_model(model_dir)
else:
    tf_model = make_deng_tf_stucture()
    # tf_model = load_weights(tf_model, weights_path)
    if SAVE_MODEL:
        tf_model.save(model_dir)

# Load Cached Test Data
if osp.exists(cache_file):
    with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
    print('data is loaded from {}'.format(cache_file))
len(roidb)

# Test Network
test_net(tf_model, roidb)






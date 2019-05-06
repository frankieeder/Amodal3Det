from model import *
from weights import *
from test import *
import os.path as osp
import pickle
from keras.models import load_model

SAVE_MODEL = False
LOAD_MODEL = False # TODO: Get this to work
SHOW_MODEL = True
model_dir = "deng_tensorflow.h5"
cache_file = "roidb_test_19_smol.pkl"
weights_path = "../pretrained/output/rgbd_det_iter_40000.h5"


# Define Model Structure
if LOAD_MODEL:
    tf_model = load_model(model_dir)
else:
    print("Making Structure")
    tf_model = make_deng_tf_stucture()
    print("Loading Weights")
    #tf_model = load_weights(tf_model, weights_path)
    if SAVE_MODEL:
        tf_model.save(model_dir)

if SHOW_MODEL:
    print("Model Summary:")
    print(tf_model.summary())
    plot_model(tf_model, show_shapes=True, show_layer_names=True, to_file='model.png')

# Load Cached Test Data
if osp.exists(cache_file):
    with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
    print('data is loaded from {}'.format(cache_file))
print(len(roidb))

# Test Network
test_net(tf_model, roidb)






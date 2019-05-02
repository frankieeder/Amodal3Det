from model import *
from weights import *

tf_model = make_deng_tf_stucture()
tf_model = load_weights(tf_model)
if True:
    out_dir = "deng_tensorflow.h5"
    tf_model.save(out_dir)





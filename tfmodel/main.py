import os.path as osp
import pickle
import model
import weights
import test

def get_tf_implementation(cache_file, weights_path, all_layer_outs=False):
    # Define Model Structure
    print("Making Structure")
    tf_model = model.make_deng_tf_test(verbose=True, all_layer_outs=all_layer_outs)
    print("Loading Weights")
    tf_model = weights.load_weights(tf_model, weights_path, verbose=True)
    return tf_model

def test_tf_implementation(cache_file, weights_path, all_layer_outs=False):
    tf_model = get_tf_implementation(cache_file, weights_path, all_layer_outs)
    model.show_model_info(tf_model)

    # Load Cached Test Data
    print("Loading Test Data")
    if osp.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = pickle.load(fid)
        print('data is loaded from {}'.format(cache_file))
    print(f"Number of Images to test: {len(roidb)}")

    # Test Network
    test.test_net(tf_model, roidb)

if __name__ == "__main__":
    test_tf_implementation(
        cache_file="roidb_test_19_smol.pkl",
        weights_path="../pretrained/output/rgbd_det_iter_40000.h5"
    )
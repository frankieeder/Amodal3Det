### NEEDS TO BE TEMPORARILY RUN IN PYTHON 2.7
import cPickle
import numpy as np

pkl = "layer_wise_outs0.pkl"
with open(pkl, 'rb') as fid:
    data = cPickle.load(fid)

for name, mat in data.items():
    print("Saving " + name)
    np.save("layer_wise_outs0/" + name + ".npy", mat)

x = 2
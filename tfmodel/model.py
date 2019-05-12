from keras.layers import Flatten, Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, ZeroPadding2D, \
    BatchNormalization, Activation, Lambda
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.utils import plot_model
import numpy as np
import math

class RoiPoolingConvSingle(Layer):
    def __init__(self, pool_size, scale_factor=1.0, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.scale_factor = scale_factor

        super(RoiPoolingConvSingle, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return (None,) + self.pool_size + (self.nb_channels,)

    def call(self, x, mask=None):
        assert (len(x) == 2)
        img = x[0]
        roi = x[1][0]  # Use only first ROI, tensorflow will discard the rest anyways

        #Misc. variables used in Caffe Implementation
        channels_ = img.shape[3]  # TODO: Might not be right
        pooled_height_ = self.pool_size[0] # TODO: pool size axis order might be wrong, but fine for our uses.
        pooled_width_ = self.pool_size[1]
        height_, width_ = img.shape #TODO: Might be wrong order here, will likely cause errors

        roi_start_w = K.cast(tf.math.round(roi[1]), 'int32') * self.scale_factor
        roi_start_h = K.cast(tf.math.round(roi[2]), 'int32') * self.scale_factor
        roi_end_w = K.cast(tf.math.round(roi[3]), 'int32') * self.scale_factor
        roi_end_h = K.cast(tf.math.round(roi[4]), 'int32') * self.scale_factor

        roi_height = tf.max(roi_end_h - roi_start_h + 1, 1)
        roi_width = tf.max(roi_end_w - roi_start_w + 1, 1)

        bin_size_h = roi_height / pooled_height_
        bin_size_w = roi_width / pooled_width_

        top_data = np.zeros((1, height_, width_, channels_))  # TODO: Is this the right shape?
        argmax_data = {}
        for c in range(channels_):
            for ph in range(pooled_height_):
                for pw in range(pooled_width_):
                    hstart = tf.floor(ph * bin_size_h)
                    wstart = tf.floor(pw * bin_size_w)
                    hend = tf.ceil((ph + 1) * bin_size_h)
                    wend = tf.ceil((pw + 1) * bin_size_w)

                    hstart = tf.min(tf.max(hstart + roi_start_h, 0), height_)
                    hend = tf.min(tf.max(hend + roi_start_h, 0), height_)
                    wstart = tf.min(tf.max(wstart + roi_start_w, 0), width_)
                    wend = tf.min(tf.max(wend + roi_start_w, 0), width_)

                    is_empty = (hend <= hstart) or (wend <= wstart)
                    # pool_index = ph * pooled_width_ + pw; TODO: is this the right axis order
                    if is_empty:
                        top_data[0, ph, pw, c] = 0
                        argmax_data[0, ph, pw, c] = -1

                    for h in range(hstart, hend):
                        for w in range(wstart, wend):
                            # index = h * width_ + w; [h, w]  TODO: is this the right axis order
                            if img[0, h, w, c] > top_data[0, ph, pw, c]:
                                top_data[0, ph, pw, c] = img[0, h, w, c]
                                argmax_data[(0, ph, pw, c)] = (0, h, w, c)
        final_output = K.cast(top_data, 'float32')
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size}
        base_config = super(RoiPoolingConvSingle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def ROIPoolingFun(pool_size, scale_factor=1.0):
    def ROIPooler(x):
        assert (len(x) == 2)
        img = x[0]
        roi = x[1][0]  # Use only first ROI, tensorflow will discard the rest anyways

        # Misc. variables used in Caffe Implementation
        channels_ = img.shape[3]  # TODO: Might not be right
        pooled_height_ = pool_size[0]  # TODO: pool size axis order might be wrong, but fine for our uses.
        pooled_width_ = pool_size[1]
        _, height_, width_, _ = img.shape  # TODO: Might be wrong order here, will likely cause errors

        roi_start_w = round(roi[1]) * scale_factor
        roi_start_h = round(roi[2]) * scale_factor
        roi_end_w = round(roi[3]) * scale_factor
        roi_end_h = round(roi[4]) * scale_factor

        roi_height = max(roi_end_h - roi_start_h + 1, 1)
        roi_width = max(roi_end_w - roi_start_w + 1, 1)

        bin_size_h = roi_height / pooled_height_
        bin_size_w = roi_width / pooled_width_

        top_data = np.zeros((1, height_, width_, channels_))  # TODO: Is this the right shape?
        argmax_data = {}
        for c in range(channels_):
            for ph in range(pooled_height_):
                for pw in range(pooled_width_):
                    hstart = math.floor(ph * bin_size_h)
                    wstart = math.floor(pw * bin_size_w)
                    hend = math.ceil((ph + 1) * bin_size_h)
                    wend = math.ceil((pw + 1) * bin_size_w)

                    hstart = min(max(hstart + roi_start_h, 0), height_)
                    hend = min(max(hend + roi_start_h, 0), height_)
                    wstart = min(max(wstart + roi_start_w, 0), width_)
                    wend = min(max(wend + roi_start_w, 0), width_)

                    is_empty = (hend <= hstart) or (wend <= wstart)
                    # pool_index = ph * pooled_width_ + pw; TODO: is this the right axis order
                    if is_empty:
                        top_data[0, ph, pw, c] = 0
                        argmax_data[0, ph, pw, c] = -1

                    for h in range(hstart, hend):
                        for w in range(wstart, wend):
                            # index = h * width_ + w; [h, w]  TODO: is this the right axis order
                            if img[0, h, w, c] > top_data[0, ph, pw, c]:
                                top_data[0, ph, pw, c] = img[0, h, w, c]
                                argmax_data[(0, ph, pw, c)] = (0, h, w, c)
        final_output = K.cast(top_data, 'float32')
        return final_output
    return ROIPooler

def ROIPoolingConv(**kwargs):
    ROIPoolingConv = Lambda(
        function=ROIPoolingFun(
            pool_size=(7, 7),
            scale_factor=(1 / 16)
        ),
        **kwargs
    )
    return ROIPoolingConv

def VGG_16_Conv(num_convs, input_tensor, suffix=''):
    padding = ZeroPadding2D(padding=(1, 1), input_shape=(224, 224, 3), name='padding' + suffix)(input_tensor)
    conv = Conv2D(num_convs, (3, 3), activation='relu', name='conv' + suffix)(padding)
    return conv

def VGG_16_Conv_BN(num_layers, input_tensor, suffix=''):
    padding3_1 = ZeroPadding2D((1, 1), name='padding' + suffix)(input_tensor)
    conv3_1 = Conv2D(num_layers, (3, 3), name='conv' + suffix)(padding3_1)
    bn_conv3_1 = BatchNormalization(epsilon=0.00001, name='bn_conv' + suffix)(conv3_1)
    relu_conv3_1 = Activation('relu', name="relu" + suffix)(bn_conv3_1)
    return relu_conv3_1

def VGG_16(img, suffix):
    conv1_1 = VGG_16_Conv(64, img, "1_1" + suffix)
    conv1_2 = VGG_16_Conv(64, conv1_1, "1_2" + suffix)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1' + suffix)(conv1_2)

    conv2_1 = VGG_16_Conv(128, pool1, "2_1" + suffix)
    conv2_2 = VGG_16_Conv(128, conv2_1, "2_2" + suffix)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2' + suffix)(conv2_2)

    conv3_1 = VGG_16_Conv_BN(256, pool2, "3_1" + suffix)
    conv3_2 = VGG_16_Conv_BN(256, conv3_1, "3_2" + suffix)
    conv3_3 = VGG_16_Conv_BN(256, conv3_2, "3_3" + suffix)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3' + suffix)(conv3_3)

    conv4_1 = VGG_16_Conv_BN(512, pool3, "4_1" + suffix)
    conv4_2 = VGG_16_Conv_BN(512, conv4_1, "4_2" + suffix)
    conv4_3 = VGG_16_Conv_BN(512, conv4_2, "4_3" + suffix)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4' + suffix)(conv4_3)

    conv5_1 = VGG_16_Conv_BN(512, pool4, "5_1" + suffix)
    conv5_2 = VGG_16_Conv_BN(512, conv5_1, "5_2" + suffix)
    conv5_3 = VGG_16_Conv_BN(512, conv5_2, "5_3" + suffix)

    return conv5_3

def VGG_16_RGB(img):
    return VGG_16(img, "")

def VGG_16_D(dmap):
    return VGG_16(dmap, "d")

def make_deng_tf_stucture(verbose=False):
    ## Inputs
    if verbose:
        print("Defining Inputs...")
    img = Input(shape=(427, 561, 3), name='img')
    dmap = Input(shape=(427, 561, 3), name='dmap')
    rois = Input(shape=(5,), name='rois')
    rois_context = Input(shape=(5,), name='rois_context')

    ## VGG-16s
    if verbose:
        print("Defining VGG-16-RGB...")
    conv5_3 = VGG_16_RGB(img)
    if verbose:
        print("Defining VGG-16-D...")
    conv5_3d = VGG_16_D(dmap)

    #### ROI Pooling
    if verbose:
        print("Defining ROI Pooling layers...")
    pool5 = ROIPoolingConv(name="pool5")([conv5_3, rois])
    pool5_context = ROIPoolingConv(name="pool5_context")([conv5_3, rois_context])
    pool5d = ROIPoolingConv(name="pool5d")([conv5_3d, rois])
    pool5d_context = ROIPoolingConv(name="pool5d_context")([conv5_3d, rois_context])

    ## Flatten
    if verbose:
        print("Defining Flatten layers...")
    flatten = Flatten(name='flatten')(pool5)
    flatten_d = Flatten(name='flatten_d')(pool5d)
    flatten_context = Flatten(name='flatten_context')(pool5_context)
    flatten_d_context = Flatten(name='flatten_d_context')(pool5d_context)

    #### Concatenation
    if verbose:
        print("Defining Concatenation layer...")
    roi_pool_rgbd = Concatenate(name='concat', axis=1)([  # TODO: Check shape of inputs and ensure correct concatenation direction.
        flatten,
        flatten_d,
        flatten_context,
        flatten_d_context
    ])

    if verbose:
        print("Defining FC layers...")
    #### FC Layers
    fc6 = Dense(
        units=4096,
        activation='relu',
        name='fc6'
    )(roi_pool_rgbd)
    drop6 = Dropout(rate=0.5, seed=42, name='drop6')(fc6)  # TODO: These might not be necessary, but the scale factor at test time compared to Caffe model seems to be causing an issue
    fc7 = Dense(
        units=4096,
        activation='relu',
        name='fc7'
    )(drop6)
    drop7 = Dropout(rate=0.5, seed=42, name="drop7")(fc7)

    if verbose:
        print("Defining Outputs...")
    #### Outputs
    cls_score = Dense(
        units=20,
        activation='softmax',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer='zeros',
        name='cls_score'
    )(drop7)
    bbox_pred_3d = Dense(
        units=140,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
        bias_initializer='zeros',
        name='bbox_pred_3d'
    )(drop7)

    ## Model Setup
    if verbose:
        print("Defining Model Object...")
    tf_model = Model(
        inputs=[img, dmap, rois, rois_context],
        outputs=[cls_score, bbox_pred_3d]
    )

    return tf_model

def make_deng_tf_test(verbose=False, all_layer_outs=False):
    tf_model = make_deng_tf_stucture(verbose)
    if all_layer_outs:
        all_layers = [layer for layer in tf_model.layers if
                      not any(n in layer.name for n in ['img', 'dmap', 'rois', 'rois_context', 'padding'])]
        all_outs = [l.output for l in all_layers]
        tf_model = Model(
            inputs=tf_model.inputs,
            outputs=all_outs
        )
    return tf_model

def show_model_info(model):
    print("Model Summary:")
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
    tf_model = make_deng_tf_test()
    show_model_info(tf_model)

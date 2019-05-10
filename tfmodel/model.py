import cv2
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, ZeroPadding2D, Lambda, BatchNormalization, Activation
from keras.models import Model
from keras.utils import plot_model
from keras.utils.conv_utils import convert_kernel
from keras.optimizers import SGD
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from tensorflow.python import debug as tf_debug


class RoiPoolingConv(Layer):
    def __init__(self, pool_size, scale_factor=1.0, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.scale_factor = scale_factor

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return (None,) + self.pool_size + (self.nb_channels,)

    def call(self, x, mask=None):
        assert (len(x) == 2)
        img = x[0]
        rois = x[1]

        def roi_cords_to_pooled(roi):
            roi *= self.scale_factor
            x_min = K.cast(tf.math.round(roi[1]), 'int32')
            y_min = K.cast(tf.math.round(roi[2]), 'int32')
            x_max = K.cast(tf.math.round(roi[3]), 'int32')
            y_max = K.cast(tf.math.round(roi[4]), 'int32')

            # Resized roi of the image to pooling size (7x7)
            section = img[:, y_min:y_max+1, x_min:x_max+1, :]  # TODO: Axis order might be backwards
            rs = tf.image.resize_images(section, self.pool_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # TODO: This doesn't seem to be max pooling...
            return rs

        final_output = tf.map_fn(
            fn=roi_cords_to_pooled,
            elems=rois
        )
        final_output = final_output[0, :, :, :, :]  # TODO: Might be keeping the wrong axis
        final_output = tf.transpose(final_output, (0, 3, 1, 2))
        final_output = K.cast(final_output, 'float32')

        #final_output = K.constant(0.0, shape=self.pool_size + (self.nb_channels,))  # Null output for testing
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

def ROI_Pool_Keras(**kwargs):
    return Lambda(lambda l: ROI_Pool_TF(*l), **kwargs)

def ROI_Pool_TF(img, rois, pool_size=(7, 7), scale_factor=(1/16)):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        DEPRECATED: num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(num_rois, 4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(num_rois, channels, pool_size, pool_size)`
    '''
    rois_norm = K.cast(rois * scale_factor, 'int32')
    rois_norm = K.cast(rois_norm, 'float32')
    def roi_cords_to_pooled(roi):
        x = K.cast(roi[1], 'int32')
        y = K.cast(roi[2], 'int32')
        w = K.cast(roi[3], 'int32') - x
        h = K.cast(roi[4], 'int32') - y

        # Resized roi of the image to pooling size (7x7)
        section = tf.image.crop_to_bounding_box(img, y, x, h+1, w+1)  # TODO: Axis order might be backwards
        rs = tf.image.resize_images(section, pool_size)  # TODO: This doesn't seem to be max pooling...
        return rs

    final_output = tf.map_fn(
        fn=roi_cords_to_pooled,
        elems=rois_norm
    )
    final_output = final_output[:, 0, :, :, :]  # TODO: Might be keeping the wrong axis
    final_output = K.cast(final_output, 'float32')
    return final_output

def ROI_Pool_For_Lambda(pool_size=(7, 7), scale_factor=(1/16)):
    pass

def make_deng_tf_stucture():
    sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

    ## Inputs
    img = Input(shape=(427, 561, 3), name='img')
    dmap = Input(shape=(427, 561, 3), name='dmap')
    rois = Input(shape=(5,), name='rois')
    rois_context = Input(shape=(5,), name='rois_context')

    conv5_3 = VGG_16_RGB(img)
    conv5_3d = VGG_16_D(dmap)


    pooling_regions = 7 #Commonly set as 7
    num_rois = 1   #Commonly set as 4


    #### ROI Pooling
    pool5 = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5"
    )([conv5_3, rois])
    pool5_context = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5_context"
    )([conv5_3, rois_context])
    pool5d = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5d"
    )([conv5_3d, rois])
    pool5d_context = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5d_context"
    )([conv5_3d, rois_context])

    #### Flatten
    flatten = Flatten(name='flatten')(pool5)
    flatten_d = Flatten(name='flatten_d')(pool5d)
    flatten_context = Flatten(name='flatten_context')(pool5_context)
    flatten_d_context = Flatten(name='flatten_d_context')(pool5d_context)

    #### Concatenation
    roi_pool_rgbd = Concatenate(name='concat', axis=1)([ #TODO: Check shape of inputs and ensure correct concatenation direction.
        flatten,
        flatten_d,
        flatten_context,
        flatten_d_context
    ])

    #### FC Layers
    fc6 = Dense(
        units=4096,
        activation='relu',
        #kernel_initializer='glorot_normal',
        #bias_initializer='constant',
        name='fc6'
    )(roi_pool_rgbd) # TODO: Do we need the learning parameters here?
    # No since we are just training right?
    # TODO: What do the dropout layers to here?
    fc7 = Dense(
        units=4096,
        activation='relu',
        #kernel_initializer='glorot_normal',
        #bias_initializer='constant',
        name='fc7'
    )(fc6) # TODO: Do we need the learning parameters here?
    # TODO: What to the dropout layers to here?

    #### Outputs
    cls_score = Dense(
        units=20,
        activation='softmax', # TODO: Is this all I need to do?
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer='zeros',
        name='cls_score'
    )(fc7)
    bbox_pred_3d = Dense(
        units=140,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
        bias_initializer='zeros',
        name='bbox_pred_3d'
    )(fc7)

    ## Model Setup
    ALL_LAYERS = True
    ins = [img, dmap, rois, rois_context]
    if ALL_LAYERS:
        tf_model = Model(
            inputs=ins,
            outputs=[cls_score, bbox_pred_3d]
        )
        all_layers = [layer for layer in tf_model.layers if not any(n in layer.name for n in ['img', 'dmap', 'rois', 'rois_context', 'padding'])]
        all_outs = [l.output for l in all_layers]
        reference_names = [(i, l.name) for i, l in enumerate(all_layers)]
        print(f"Layer names by output, for reference:\n{reference_names}")
        tf_model = Model(
            inputs=ins,
            outputs=all_outs
        )
    else:
        tf_model = Model(
            inputs=ins,
            outputs=[cls_score, bbox_pred_3d]
        )
    return tf_model









import cv2
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, ZeroPadding2D, Lambda, BatchNormalization
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
        tf.print(img)
        rois = x[1]
        tf.print(rois)
        rois_norm = K.cast(rois * self.scale_factor, 'int32')
        rois_norm = K.cast(rois_norm, 'float32')

        def roi_cords_to_pooled(roi):
            x = K.cast(roi[0], 'int32')
            y = K.cast(roi[1], 'int32')
            w = K.cast(roi[2], 'int32') - x
            h = K.cast(roi[3], 'int32') - y

            # Resized roi of the image to pooling size (7x7)
            section = tf.image.crop_to_bounding_box(img, y, x, h+1, w+1)  # TODO: Axis order might be backwards
            rs = tf.image.resize_images(section, self.pool_size)  # TODO: This doesn't seem to be max pooling...
            return rs

        final_output = tf.map_fn(
            fn=roi_cords_to_pooled,
            elems=rois_norm
        )
        final_output = final_output[0, :, :, :, :]  # TODO: Might be keeping the wrong axis
        tf.print()
        final_output = K.cast(final_output, 'float32')

        #final_output = K.constant(0.0, shape=self.pool_size + (self.nb_channels,))  # Null output for testing
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def VGG_16_RGB(img):
    padding1_1 = ZeroPadding2D(padding=(1, 1), input_shape=(224, 224, 3), name='padding1_1')(img)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(padding1_1)
    padding1_2 = ZeroPadding2D((1, 1), name='padding1_2')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(padding1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    padding2_1 = ZeroPadding2D((1, 1), name='padding2_1')(pool1)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(padding2_1)
    padding2_2 = ZeroPadding2D((1, 1), name='padding2_2')(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(padding2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    padding3_1 = ZeroPadding2D((1, 1), name='padding3_1')(pool2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(padding3_1)
    bn_conv3_1 = BatchNormalization(epsilon=0.00001, name='bn_conv3_1')(conv3_1)
    padding3_2 = ZeroPadding2D((1, 1), name='padding3_2')(bn_conv3_1)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(padding3_2)
    bn_conv3_2 = BatchNormalization(epsilon=0.00001,  name='bn_conv3_2')(conv3_2)
    padding3_3 = ZeroPadding2D((1, 1), name='padding3_3')(bn_conv3_2)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(padding3_3)
    bn_conv3_3 = BatchNormalization(epsilon=0.00001,  name='bn_conv3_3')(conv3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(bn_conv3_3)

    padding4_1 = ZeroPadding2D((1, 1), name='padding4_1')(pool3)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(padding4_1)
    bn_conv4_1 = BatchNormalization(epsilon=0.00001,  name='bn_conv4_1')(conv4_1)
    padding4_2 = ZeroPadding2D((1, 1), name='padding4_2')(bn_conv4_1)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(padding4_2)
    bn_conv4_2 = BatchNormalization(epsilon=0.00001,  name='bn_conv4_2')(conv4_2)
    padding4_3 = ZeroPadding2D((1, 1), name='padding4_3')(bn_conv4_2)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(padding4_3)
    bn_conv4_3 = BatchNormalization(epsilon=0.00001,  name='bn_conv4_3')(conv4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(bn_conv4_3)

    padding5_1 = ZeroPadding2D((1, 1), name='padding5_1')(pool4)
    conv5_1 = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(padding5_1)
    bn_conv5_1 = BatchNormalization(epsilon=0.00001,  name='bn_conv5_1')(conv5_1)
    padding5_2 = ZeroPadding2D((1, 1), name='padding5_2')(bn_conv5_1)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(padding5_2)
    bn_conv5_2 = BatchNormalization(epsilon=0.00001,  name='bn_conv5_2')(conv5_2)
    padding5_3 = ZeroPadding2D((1, 1), name='padding5_3')(bn_conv5_2)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(padding5_3)
    bn_conv5_3 = BatchNormalization(epsilon=0.00001,  name='bn_conv5_3')(conv5_3)

    return bn_conv5_3


def VGG_16_D(dmap):
    padding1_1d = ZeroPadding2D(padding=(1, 1), input_shape=(224, 224, 3), name='padding1_1d')(dmap)
    conv1_1d = Conv2D(64, (3, 3), activation='relu', name='conv1_1d')(padding1_1d)
    padding1_2d = ZeroPadding2D((1, 1), name='padding1_2d')(conv1_1d)
    conv1_2d = Conv2D(64, (3, 3), activation='relu', name='conv1_2d')(padding1_2d)
    pool1d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1d')(conv1_2d)

    padding2_1d = ZeroPadding2D((1, 1), name='padding2_1d')(pool1d)
    conv2_1d = Conv2D(128, (3, 3), activation='relu', name='conv2_1d')(padding2_1d)
    padding2_2d = ZeroPadding2D((1, 1), name='padding2_2d')(conv2_1d)
    conv2_2d = Conv2D(128, (3, 3), activation='relu', name='conv2_2d')(padding2_2d)
    pool2d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2d')(conv2_2d)

    padding3_1d = ZeroPadding2D((1, 1), name='padding3_1d')(pool2d)
    conv3_1d = Conv2D(256, (3, 3), activation='relu', name='conv3_1d')(padding3_1d)
    bn_conv3_1d = BatchNormalization(epsilon=0.00001,  name='bn_conv3_1d')(conv3_1d)
    padding3_2d = ZeroPadding2D((1, 1), name='padding3_2d')(bn_conv3_1d)
    conv3_2d = Conv2D(256, (3, 3), activation='relu', name='conv3_2d')(padding3_2d)
    bn_conv3_2d = BatchNormalization(epsilon=0.00001,  name='bn_conv3_2d')(conv3_2d)
    padding3_3d = ZeroPadding2D((1, 1), name='padding3_3d')(bn_conv3_2d)
    conv3_3d = Conv2D(256, (3, 3), activation='relu', name='conv3_3d')(padding3_3d)
    bn_conv3_3d = BatchNormalization(epsilon=0.00001,  name='bn_conv3_3d')(conv3_3d)
    pool3d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3d')(bn_conv3_3d)

    padding4_1d = ZeroPadding2D((1, 1), name='padding4_1d')(pool3d)
    conv4_1d = Conv2D(512, (3, 3), activation='relu', name='conv4_1d')(padding4_1d)
    bn_conv4_1d = BatchNormalization(epsilon=0.00001,  name='bn_conv4_1d')(conv4_1d)
    padding4_2d = ZeroPadding2D((1, 1), name='padding4_2d')(bn_conv4_1d)
    conv4_2d = Conv2D(512, (3, 3), activation='relu', name='conv4_2d')(padding4_2d)
    bn_conv4_2d = BatchNormalization(epsilon=0.00001,  name='bn_conv4_2d')(conv4_2d)
    padding4_3d = ZeroPadding2D((1, 1), name='padding4_3d')(bn_conv4_2d)
    conv4_3d = Conv2D(512, (3, 3), activation='relu', name='conv4_3d')(padding4_3d)
    bn_conv4_3d = BatchNormalization(epsilon=0.00001,  name='bn_conv4_3d')(conv4_3d)
    pool4d = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4d')(bn_conv4_3d)

    padding5_1d = ZeroPadding2D((1, 1), name='padding5_1d')(pool4d)
    conv5_1d = Conv2D(512, (3, 3), activation='relu', name='conv5_1d')(padding5_1d)
    bn_conv5_1d = BatchNormalization(epsilon=0.00001,  name='bn_conv5_1d')(conv5_1d)
    padding5_2d = ZeroPadding2D((1, 1), name='padding5_2d')(bn_conv5_1d)
    conv5_2d = Conv2D(512, (3, 3), activation='relu', name='conv5_2d')(padding5_2d)
    bn_conv5_2d = BatchNormalization(epsilon=0.00001,  name='bn_conv5_2d')(conv5_2d)
    padding5_3d = ZeroPadding2D((1, 1), name='padding5_3d')(bn_conv5_2d)
    conv5_3d = Conv2D(512, (3, 3), activation='relu', name='conv5_3d')(padding5_3d)
    bn_conv5_3d = BatchNormalization(epsilon=0.00001,  name='bn_conv5_3d')(conv5_3d)

    return bn_conv5_3d

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
        x = K.cast(roi[0], 'int32')
        y = K.cast(roi[1], 'int32')
        w = K.cast(roi[2], 'int32') - x
        h = K.cast(roi[3], 'int32') - y

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

    bn_conv5_3 = VGG_16_RGB(img)
    bn_conv5_3d = VGG_16_D(dmap)


    pooling_regions = 7 #Commonly set as 7
    num_rois = 1   #Commonly set as 4


    #### ROI Pooling
    pool5 = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5"
    )([bn_conv5_3, rois])
    pool5_context = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5_context"
    )([bn_conv5_3, rois_context])
    pool5d = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5d"
    )([bn_conv5_3d, rois])
    pool5d_context = RoiPoolingConv(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5d_context"
    )([bn_conv5_3d, rois_context])
    """pool5 = RoiPoolingConv(
        pooling_regions,
        num_rois,
        name="pool5"
    )([bn_conv5_3, rois])
    pool5_context = RoiPoolingConv(
        pooling_regions,
        num_rois,
        name="pool5_context"
    )([bn_conv5_3, rois_context])
    pool5d = RoiPoolingConv(
        pooling_regions,
        num_rois,
        name="pool5d"
    )([bn_conv5_3d, rois])
    pool5d_context = RoiPoolingConv(
        pooling_regions,
        num_rois,
        name="pool5d_context"
    )([bn_conv5_3d, rois_context])"""

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
        kernel_initializer='glorot_normal',
        bias_initializer='constant',
        name='fc6'
    )(roi_pool_rgbd) # TODO: Do we need the learning parameters here?
    # No since we are just training right?
    # TODO: What do the dropout layers to here?
    fc7 = Dense(
        units=4096,
        activation='relu',
        kernel_initializer='glorot_normal',
        bias_initializer='constant',
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










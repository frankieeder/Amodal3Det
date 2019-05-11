from keras.layers import Flatten, Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, ZeroPadding2D, \
    BatchNormalization, Activation
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.utils import plot_model

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
        roi = x[1]

        roi *= self.scale_factor
        x_min = K.cast(tf.math.round(roi[0, 1]), 'int32')
        y_min = K.cast(tf.math.round(roi[0, 2]), 'int32')
        x_max = K.cast(tf.math.round(roi[0, 3]), 'int32')
        y_max = K.cast(tf.math.round(roi[0, 4]), 'int32')

        section = img[:, y_min:y_max+1, x_min:x_max+1, :]
        # TODO: This doesn't seem to be max pooling...
        rs = tf.image.resize_images(section, self.pool_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        #rs = rs[0, :, :, :, :]
        rs = tf.transpose(rs, (0, 3, 1, 2))
        final_output = K.cast(rs, 'float32')
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size}
        base_config = super(RoiPoolingConvSingle, self).get_config()
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
    pool5 = RoiPoolingConvSingle(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5"
    )([conv5_3, rois])
    pool5_context = RoiPoolingConvSingle(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5_context"
    )([conv5_3, rois_context])
    pool5d = RoiPoolingConvSingle(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5d"
    )([conv5_3d, rois])
    pool5d_context = RoiPoolingConvSingle(
        pool_size=(7, 7),
        scale_factor=(1 / 16),
        name="pool5d_context"
    )([conv5_3d, rois_context])

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
                      not any(n in layer.name for n in ['img', 'dmap', 'rois', 'rois_context'])]
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

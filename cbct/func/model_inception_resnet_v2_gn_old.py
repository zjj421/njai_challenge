import os

from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Concatenate, UpSampling2D, Activation
from keras.models import Model
from keras_applications import get_keras_submodule
from keras_applications.inception_resnet_v2 import inception_resnet_block, InceptionResNetV2
from tqdm import tqdm

from func.config import Config
from func.group_norm import GroupNormalization

layers = get_keras_submodule('layers')
backend = get_keras_submodule('backend')
CONFIG = Config()
GN_AXIS = 3
CHANNEL_AXIS = GN_AXIS


def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = GroupNormalization(name=name, axis=GN_AXIS)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv


def conv2d_gn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + GN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_gn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `GroupNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        gn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        gn_name = None if name is None else name + '_gn'
        # try:
        #     x = GroupNormalization(axis=gn_axis, groups=32,
        #                            scale=False,
        #                            name=gn_name)(x)
        # except:
        x = GroupNormalization(axis=gn_axis, groups=filters // 4,
                               scale=False,
                               name=gn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def get_inception_resnet_v2_unet_sigmoid_gn(input_shape=(CONFIG.img_h, CONFIG.img_w, CONFIG.img_c),
                                            output_channels=1,
                                            weights='imagenet'):
    inp = Input(input_shape)

    # Stem block: 35 x 35 x 192
    x = conv2d_gn(inp, 32, 3, strides=2, padding='same')
    x = conv2d_gn(x, 32, 3, padding='same')
    x = conv2d_gn(x, 64, 3)
    conv1 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = conv2d_gn(x, 80, 1, padding='same')
    x = conv2d_gn(x, 192, 3, padding='same')
    conv2 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_gn(x, 96, 1)
    branch_1 = conv2d_gn(x, 48, 1)
    branch_1 = conv2d_gn(branch_1, 64, 5)
    branch_2 = conv2d_gn(x, 64, 1)
    branch_2 = conv2d_gn(branch_2, 96, 3)
    branch_2 = conv2d_gn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_gn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)
    conv3 = x
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_gn(x, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_gn(x, 256, 1)
    branch_1 = conv2d_gn(branch_1, 256, 3)
    branch_1 = conv2d_gn(branch_1, 384, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)
    conv4 = x
    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_gn(x, 256, 1)
    branch_0 = conv2d_gn(branch_0, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_gn(x, 256, 1)
    branch_1 = conv2d_gn(branch_1, 288, 3, strides=2, padding='same')
    branch_2 = conv2d_gn(x, 256, 1)
    branch_2 = conv2d_gn(branch_2, 288, 3)
    branch_2 = conv2d_gn(branch_2, 320, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_gn(x, 1536, 1, name='conv_7b')
    conv5 = x

    conv6 = conv_block(UpSampling2D()(conv5), 320)
    conv6 = concatenate([conv6, conv4], axis=-1)
    conv6 = conv_block(conv6, 320)

    conv7 = conv_block(UpSampling2D()(conv6), 256)
    conv7 = concatenate([conv7, conv3], axis=-1)
    conv7 = conv_block(conv7, 256)

    conv8 = conv_block(UpSampling2D()(conv7), 128)
    conv8 = concatenate([conv8, conv2], axis=-1)
    conv8 = conv_block(conv8, 128)

    conv9 = conv_block(UpSampling2D()(conv8), 96)
    conv9 = concatenate([conv9, conv1], axis=-1)
    conv9 = conv_block(conv9, 96)

    conv10 = conv_block(UpSampling2D()(conv9), 64)
    conv10 = conv_block(conv10, 64)
    res = Conv2D(output_channels, (1, 1), activation='sigmoid')(conv10)

    model = Model(inp, res)

    if weights == 'imagenet':
        inception_resnet_v2 = InceptionResNetV2(weights=weights, include_top=False,
                                                input_shape=(input_shape[0], input_shape[1], 3))
        print("Loading imagenet weights ...")
        for i in tqdm(range(2, len(inception_resnet_v2.layers) - 1)):
            model.layers[i].set_weights(inception_resnet_v2.layers[i].get_weights())
            model.layers[i].trainable = False
        print("imagenet weights have been loaded.")

    return model


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # model = InceptionResNetV2(weights="imagenet", include_top=False,
    #                                             input_shape=(224, 224, 3))
    # model.summary()
    # exit()

    # model = get_inception_resnet_v2_unet_sigmoid(weights=None)
    # model.summary()
    # plot_model(model, to_file="/tmp/inception_resnet_v2_unet.png", show_shapes=True,
    #            )
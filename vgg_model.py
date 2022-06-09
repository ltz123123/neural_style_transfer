from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model


def build_model(pooling="avg"):
    img_input = Input(shape=(None, None, 3))

    # Block 1
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    if pooling == "avg":
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    elif pooling == "max":
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)
    if pooling == "avg":
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    elif pooling == "max":
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv4')(x)
    if pooling == "avg":
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    elif pooling == "max":
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv4')(x)
    if pooling == "avg":
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    elif pooling == "max":
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv4')(x)
    if pooling == "avg":
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    elif pooling == "max":
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = GlobalMaxPooling2D()(x)

    model = Model(img_input, x, name='vgg19')
    model.load_weights("vgg_weights.h5")

    return model

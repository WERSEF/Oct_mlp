# coding:utf-8
# /usr/bin/env python

"""
Author: Hanoch
Email: hewersef@gmail.com

date: 2019/5/15 9:16
desc:
"""
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import oct_conv

def _create_normal_residual_block(inputs, ch, N):
    # Conv with skip connections
    x = inputs
    for i in range(N):
        # adjust channels
        if i == 0:
            skip = layers.Conv2D(ch, 1)(x)
            skip = layers.BatchNormalization()(skip)
            skip = layers.Activation("relu")(skip)
        else:
            skip = x
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Add()([x, skip])
    return x

def _create_octconv_residual_block(inputs, ch, N, alpha):
    high, low = inputs
    # OctConv with skip connections
    for i in range(N):
        # adjust channels
        if i == 0:
            skip_high = layers.Conv2D(int(ch*(1-alpha)), 1)(high)
            skip_high = layers.BatchNormalization()(skip_high)
            skip_high = layers.Activation("relu")(skip_high)

            skip_low = layers.Conv2D(int(ch*alpha), 1)(low)
            skip_low = layers.BatchNormalization()(skip_low)
            skip_low = layers.Activation("relu")(skip_low)
        else:
            skip_high, skip_low = high, low

        high, low = oct_conv.OctConv2D(filters=ch, alpha=alpha)([high, low])
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high, low = oct_conv.OctConv2D(filters=ch, alpha=alpha)([high, low])
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high = layers.Add()([high, skip_high])
        low = layers.Add()([low, skip_low])
    return [high, low]

def _create_octconvmlp_residual_block(inputs, ch, N, alpha):
    high, low = inputs
    # OctConv with skip connections
    for i in range(N):
        # adjust channels
        if i == 0:
            skip_high = layers.Conv2D(int(ch*(1-alpha)), 1)(high)
            skip_high = layers.BatchNormalization()(skip_high)
            skip_high = layers.Activation("relu")(skip_high)

            skip_low = layers.Conv2D(int(ch*alpha), 1)(low)
            skip_low = layers.BatchNormalization()(skip_low)
            skip_low = layers.Activation("relu")(skip_low)
        else:
            skip_high, skip_low = high, low

        high, low = oct_conv.OctConv2D(filters=ch, alpha=alpha)([high, low])
        high = layers.Conv2D(int(ch*(1-alpha)), 1)(high)
        low = layers.Conv2D(int(ch*alpha), 1)(low)
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high, low = oct_conv.OctConv2D(filters=ch, alpha=alpha)([high, low])        
        high = layers.Conv2D(int(ch*(1-alpha)), 1)(high)
        low = layers.Conv2D(int(ch*alpha), 1)(low)
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high = layers.Add()([high, skip_high])
        low = layers.Add()([low, skip_low])
    return [high, low]

def create_normal_wide_resnet(N=4, k=10):
    """
    Create vanilla conv Wide ResNet (N=4, k=10)
    """
    # input
    input = layers.Input((32,32,3))
    # 16 channels block
    x = layers.Conv2D(16, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # 1st block
    x = _create_normal_residual_block(x, 16*k, N)
    # The original wide resnet is stride=2 conv for downsampling,
    # but replace them to average pooling because centers are shifted when octconv
    # 2nd block
    x = layers.AveragePooling2D(2)(x)
    x = _create_normal_residual_block(x, 32*k, N)
    # 3rd block
    x = layers.AveragePooling2D(2)(x)
    x = _create_normal_residual_block(x, 64*k, N)
    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_octconv_wide_resnet(alpha, N=4, k=10):
    """
    Create OctConv Wide ResNet(N=4, k=10)
    """
    # Input
    input = layers.Input((32,32,3))
    # downsampling for lower
    low = layers.AveragePooling2D(2)(input)

    # 16 channels block
    high, low = oct_conv.OctConv2D(filters=16, alpha=alpha)([input, low])
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)

    # 1st block
    high, low = _create_octconv_residual_block([high, low], 16*k, N, alpha)
    # 2nd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconv_residual_block([high, low], 32*k, N, alpha)
    # 3rd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconv_residual_block([high, low], 64*k, N, alpha)
    # concat
    high = layers.AveragePooling2D(2)(high)
    x = layers.Concatenate()([high, low])
    x = layers.Conv2D(64*k, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_octconvmlp_wide_resnet(alpha, N=4, k=10):
    """
    Create OctConvmlp Wide ResNet(N=4, k=10)
    """
    # Input
    input = layers.Input((32,32,3))
    # downsampling for lower
    low = layers.AveragePooling2D(2)(input)

    # 16 channels block
    high, low = oct_conv.OctConv2D(filters=16, alpha=alpha)([input, low])
    high = layers.Conv2D(int(16*(1-alpha)), 1)(input)
    low = layers.Conv2D(int(16*alpha), 1)(low)
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)

    # 1st block
    high, low = _create_octconvmlp_residual_block([high, low], 16*k, N, alpha)
    # 2nd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconvmlp_residual_block([high, low], 32*k, N, alpha)
    # 3rd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconvmlp_residual_block([high, low], 64*k, N, alpha)
    # concat
    high = layers.AveragePooling2D(2)(high)
    x = layers.Concatenate()([high, low])
    x = layers.Conv2D(64*k, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_normal_vgg():
    # input
    input = layers.Input((32,32,3))
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_octcov_vgg(alpha):
    # input
    input = layers.Input((32, 32, 3))
    # downsampling for lower
    low = layers.AveragePooling2D(2)(input)

    # Block 1
    high, low = oct_conv.OctConv2D(filters=64, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([input, low])
    high, low  = oct_conv.OctConv2D(filters=64, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low ])

    # Block 2
    high, low = oct_conv.OctConv2D(filters=128, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high, low = oct_conv.OctConv2D(filters=128, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high = layers.MaxPooling2D((2, 2), strides=(2, 2))(high)
    low = layers.MaxPooling2D((2, 2), strides=(2, 2))(low)

    # Block 3
    high, low = oct_conv.OctConv2D(filters=256, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high, low = oct_conv.OctConv2D(filters=256, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high, low = oct_conv.OctConv2D(filters=256, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high = layers.MaxPooling2D((2, 2), strides=(2, 2))(high)
    low = layers.MaxPooling2D((2, 2), strides=(2, 2))(low)

    # Block 4
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high = layers.MaxPooling2D((2, 2), strides=(2, 2))(high)
    low = layers.MaxPooling2D((2, 2), strides=(2, 2))(low)

    # Block 5
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])

    # concat
    high = layers.AveragePooling2D(2)(high)
    x = layers.Concatenate()([high, low])
    x = layers.Conv2D(4096, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_octcovmlp_vgg(alpha):
    # input
    input = layers.Input((32, 32, 3))
    # downsampling for lower
    low = layers.AveragePooling2D(2)(input)

    # Block 1
    high, low = oct_conv.OctConv2D(filters=64, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([input, low])        
    high = layers.Conv2D(int(64*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(64*alpha), 1)(low)        
    high = layers.Conv2D(int(64*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(64*alpha), 1)(low)

    high, low  = oct_conv.OctConv2D(filters=64, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low ])        
    high = layers.Conv2D(int(64*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(64*alpha), 1)(low)        
    high = layers.Conv2D(int(64*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(64*alpha), 1)(low)

    # Block 2
    high, low = oct_conv.OctConv2D(filters=128, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(128*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(128*alpha), 1)(low)        
    high = layers.Conv2D(int(128*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(128*alpha), 1)(low)
    high, low = oct_conv.OctConv2D(filters=128, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(128*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(128*alpha), 1)(low)        
    high = layers.Conv2D(int(128*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(128*alpha), 1)(low)
    high = layers.MaxPooling2D((2, 2), strides=(2, 2))(high)
    low = layers.MaxPooling2D((2, 2), strides=(2, 2))(low)

    # Block 3
    high, low = oct_conv.OctConv2D(filters=256, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(256*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(256*alpha), 1)(low)        
    high = layers.Conv2D(int(256*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(256*alpha), 1)(low)
    high, low = oct_conv.OctConv2D(filters=256, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(256*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(256*alpha), 1)(low)        
    high = layers.Conv2D(int(256*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(256*alpha), 1)(low)
    high, low = oct_conv.OctConv2D(filters=256, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(256*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(256*alpha), 1)(low)        
    high = layers.Conv2D(int(256*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(256*alpha), 1)(low)
    high = layers.MaxPooling2D((2, 2), strides=(2, 2))(high)
    low = layers.MaxPooling2D((2, 2), strides=(2, 2))(low)

    # Block 4
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)
    high = layers.MaxPooling2D((2, 2), strides=(2, 2))(high)
    low = layers.MaxPooling2D((2, 2), strides=(2, 2))(low)

    # Block 5
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)
    high, low = oct_conv.OctConv2D(filters=512, alpha=alpha, kernel_size=(3, 3), activation='relu', padding='same')([high, low])        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)        
    high = layers.Conv2D(int(512*(1-alpha)), 1)(high)
    low = layers.Conv2D(int(512*alpha), 1)(low)

    # concat
    high = layers.AveragePooling2D(2)(high)
    x = layers.Concatenate()([high, low])
    x = layers.Conv2D(4096, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

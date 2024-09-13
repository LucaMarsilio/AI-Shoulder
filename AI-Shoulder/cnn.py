'''
Convolution Neural Networks
'''

import os
from utils import PEE
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import *
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, BatchNormalization,
    Activation, Flatten, Concatenate, Dense, Dropout,
    Softmax, GlobalAveragePooling3D,
)


class CELUNet(object):
    """
    This class provides a simple interface to create
    a CELU-Net network with custom parameters. CEL-Unet has the same encoding path of simple Unet and splits
    decoding branch in two parallel paths, one for filled masks segmentation, another for boundary segmentation.
    Args:
        input_size: input_size: input size for the network. If the input of the network is two-dimensional, input size must be
        of type (input_dim_1, input_dim_2, 1). If the input is three-dimensional, then input size must be
        of type (input_dim_1, input_dim_2, input_dim_3, 1).
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        add_inception_module: boolean flag to determine if inception module should be applied in the first and last enc/dec layers 
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(None, None, None, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=5,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 add_inception_module=True,
                 kernel_regularizer=l2(0.001),
                 bias_regularizer=l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.add_inception_module = add_inception_module
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
            pee_input_size = 2
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
            pee_input_size = 3
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size, name="input_layer")
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch: First Layer
        for j in range(2):
            if self.add_inception_module == True:
                # Inception Module
                temp_layer = self.inception_module(
                    conv_layer=conv_layer,
                    input_layer=temp_layer,
                    n_filters=self.n_initial_filters,
                    first_kernel_size=self.kernel_size,
                    stride=self.strides,
                    pad=self.padding,
                    activation='linear',
                    large_kernels=True
                )
            if self.add_inception_module == False:
                # Standard Convolution
                temp_layer = conv_layer(self.n_initial_filters,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
            # Batch Normalization
            if self.add_batch_normalization:
                temp_layer = layers.BatchNormalization(
                    axis=-1, fused=False)(temp_layer)
            # Activation Layer
            temp_layer = layers.Activation(self.activation)(temp_layer)
        # Append for Skip Connection
        downsampling_layers.append(temp_layer)
        # Apply Max Pooling
        temp_layer = max_pool_layer(pool_size=self.pool_size,
                                    strides=self.pool_strides,
                                    padding=self.padding)(temp_layer)

        # Down sampling branch: Remaining Layers
        for i in range(self.depth - 1):
            for j in range(2):
                # Standard Convolution
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i + 1),
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer)(temp_layer)
                # Batch Normalization
                if self.add_batch_normalization:
                    temp_layer = layers.BatchNormalization(
                        axis=-1, fused=False)(temp_layer)
                # Activation Layer
                temp_layer = layers.Activation(self.activation)(temp_layer)
            # Append for Skip Connection
            downsampling_layers.append(temp_layer)
            # Apply Max Pooling
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding)(temp_layer)

        for j in range(2):
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters * pow(2, self.depth), kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation='linear',
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer)(temp_layer)
            if self.add_batch_normalization:
                temp_layer = layers.BatchNormalization(
                    axis=-1, fused=False)(temp_layer)
            # activation
            temp_layer = layers.Activation(self.activation)(temp_layer)

        # Up sampling branch
        temp_layer_edge = temp_layer
        temp_layer_merge = temp_layer

        for i in range(self.depth):
            # EDGE PATH
            temp_layer_edge = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_edge)
            temp_layer_edge = layers.Activation(
                self.activation)(temp_layer_edge)

            # MASK PATH
            temp_layer_mask = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            temp_layer_mask = layers.Activation(
                self.activation)(temp_layer_mask)

            # Concatenation
            temp_layer_edge = layers.Concatenate(axis=self.n_dim)(
                [downsampling_layers[(self.depth - 1) - i], temp_layer_edge])
            temp_layer_mask = layers.Concatenate(axis=self.n_dim)(
                [downsampling_layers[(self.depth - 1) - i], temp_layer_mask])

            for j in range(2):
                temp_layer_edge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_edge)
                if self.add_batch_normalization:
                    temp_layer_edge = layers.BatchNormalization(
                        axis=-1, fused=False)(temp_layer_edge)
                temp_layer_edge = layers.Activation(
                    self.activation)(temp_layer_edge)

                temp_layer_mask = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_mask)
                if self.add_batch_normalization:
                    temp_layer_mask = layers.BatchNormalization(
                        axis=-1, fused=False)(temp_layer_mask)
                temp_layer_mask = layers.Activation(
                    self.activation)(temp_layer_mask)

            temp_layer_edge = PEE(temp_layer_edge, self.n_initial_filters *
                                  pow(2, (self.depth - 1) - i), input_dims=pee_input_size)

            temp_layer_merge = Concatenate()(
                [temp_layer_edge, temp_layer_mask])
            temp_layer_merge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                          kernel_size=self.kernel_size,
                                          strides=self.strides,
                                          padding=self.padding,
                                          activation='linear',
                                          kernel_regularizer=self.kernel_regularizer,
                                          bias_regularizer=self.bias_regularizer)(temp_layer_merge)

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer_mask = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_merge)

        temp_layer_edge = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_edge)

        output_tensor_edge = Softmax(
            axis=-1, dtype='float32', name='out_edge')(temp_layer_edge)
        output_tensor_mask = Softmax(
            axis=-1, dtype='float32', name='out_mask')(temp_layer_mask)
        self.model = Model(inputs=[input_tensor], outputs=[
                           output_tensor_edge, output_tensor_mask])

    def inception_module(
            self, conv_layer, input_layer, n_filters, first_kernel_size, stride, pad, activation, large_kernels):
        '''
        Adds a 3 branch Inception module to the encoding path of a UNET like network
        Input: conv_layer         -> conv2D, conv2DTranspose, conv3D, conv3DTranspose
               input_layer        -> output of the previous network layer
               n_filters          -> number of feature maps
               first_kernel_size  -> kernel size of the first inception branch
               stride             -> stride for the conv filter
               pad                -> padding type for the conv filter
               activation         -> activation type for the conv filter
               large_kernels      -> if True  ks_1=f_k_s, ks_2=f_k_s+4, ks_3=f_k_s+8
                                     if False ks_1=f_k_s, ks_2=f_k_s+2, ks_3=f_k_s+4
        Output: output of the inception layer                                    
        '''
        if large_kernels is False:
            second_kernel_size = tuple(
                [size + 2 for size in first_kernel_size])
            third_kernel_size = tuple([size + 4 for size in first_kernel_size])
        elif large_kernels is True:
            second_kernel_size = tuple(
                [size + 4 for size in first_kernel_size])
            third_kernel_size = tuple([size + 8 for size in first_kernel_size])
        else:
            raise ValueError("large_kernels can be set to True or False.")
        # First Inception Branch
        inception_1 = conv_layer(n_filters,
                                 kernel_size=first_kernel_size,
                                 strides=stride,
                                 padding=pad,
                                 activation=activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer)(input_layer)
        # Second Inception Branch
        inception_2 = conv_layer(n_filters,
                                 kernel_size=second_kernel_size,
                                 strides=stride,
                                 padding=pad,
                                 activation=activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer)(input_layer)
        # Third Inception Branch
        inception_3 = conv_layer(n_filters,
                                 kernel_size=third_kernel_size,
                                 strides=stride,
                                 padding=pad,
                                 activation=activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer)(input_layer)
        # Layers Concatenation
        inception_output = layers.Concatenate(axis=self.n_dim)(
            [inception_1, inception_2, inception_3])
        # Return to Original Feature Maps
        inception_output = conv_layer(n_filters,
                                      kernel_size=first_kernel_size,
                                      strides=stride,
                                      padding=pad,
                                      activation=activation,
                                      kernel_regularizer=self.kernel_regularizer,
                                      bias_regularizer=self.bias_regularizer)(inception_output)

        return inception_output

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())


class ArthroNet(object):

    def __init__(self,
                 input_size=(192, 192, 192, 1),
                 kernel_size=(3, 3, 3),
                 kernel_stride=(1, 1, 1),
                 pool_size=(2, 2, 2),
                 pool_stride=(2, 2, 2),
                 consecutive_conv=1,
                 encoder_depth=6,
                 encoder_activation='relu',
                 classif_activation='relu',
                 padding='same',
                 n_initial_filters=12,
                 add_batch_normalization=True,
                 add_global_average=False,
                 drop_rate=0.5,
                 n_initial_dense_units=128,
                 classification_depth=2,
                 kernel_regularizer=l2(0.001),
                 bias_regularizer=l2(0.001),
                 n_classes=3):
        '''
        Initialization of the classification network class
        '''

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.consecutive_conv = consecutive_conv
        self.encoder_depth = encoder_depth
        self.encoder_activation = encoder_activation
        self.classif_activation = classif_activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.add_batch_normalization = add_batch_normalization
        self.add_global_average = add_global_average
        self.drop_rate = drop_rate
        self.n_initial_dense_units = n_initial_dense_units
        self.classification_depth = classification_depth
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates the classification network architecture
        '''

        # Input Layer
        input_layer = Input(shape=self.input_size, name="input_layer")

        # Define temp layer to enter the for loop
        temp_layer = input_layer

        # For loop to define each layer for the whole encoder depth
        for n_layer in range(0, self.encoder_depth):

            if n_layer < 2:
                curr_filters = self.n_initial_filters
            elif n_layer >= 2 and n_layer < 4:
                curr_filters = self.n_initial_filters * 2
            elif n_layer >= 4 and n_layer < 6:
                curr_filters = self.n_initial_filters * 4
            elif n_layer >= 6 and n_layer < 8:
                curr_filters = self.n_initial_filters * 8
            else:
                curr_filters = self.n_initial_filters * 16

            # For loop to generate each layer as consecutive sequence of Conv3D-BN-Activation
            for i in range(0, self.consecutive_conv):

                # Standard 3D Convolution
                temp_layer = Conv3D(
                    curr_filters,
                    # self.n_initial_filters * pow(2, n_layer),
                    kernel_size=self.kernel_size,
                    strides=self.kernel_stride,
                    padding=self.padding,
                    activation='linear',
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name=f'conv_{n_layer}_{i}')(temp_layer)

                # Batch Normalization
                if self.add_batch_normalization:
                    temp_layer = BatchNormalization(
                        axis=-1,
                        fused=False,
                        name=f'bn_{n_layer}_{i}')(temp_layer)

                # Activation Layer
                temp_layer = Activation(
                    self.encoder_activation,
                    name=f'activation_{n_layer}_{i}')(temp_layer)

            # Avoid MaxPooling3D if I'm at the last layer
            if n_layer == self.encoder_depth - 1:
                break

            # MaxPooling3D to reduce volume size
            temp_layer = MaxPooling3D(
                pool_size=self.pool_size,
                strides=self.pool_stride,
                padding=self.padding,
                name=f'max_pool_{n_layer}')(temp_layer)

        # Add Global Average Pooling 3D to reduce number of parameters
        if self.add_global_average:
            temp_layer = GlobalAveragePooling3D(
                name='gap')(temp_layer)
        else:
            # Dropout Layer
            temp_layer = Dropout(
                # self.drop_rate,
                0.3,
                name='dropout')(temp_layer)
            # Flatten Layer
            temp_layer = Flatten(
                name='flatten')(temp_layer)

        # Dropout Layer
        temp_layer_osteo = Dropout(
            0.6,
            name='dropout_osteo')(temp_layer)
        temp_layer_kl = Dropout(
            0.6,
            name='dropout_kl')(temp_layer)
        temp_layer_hsa = Dropout(
            0.6,
            name='dropout_hsa')(temp_layer)

        # Dense Layers
        temp_layer_osteo = Dense(
            self.n_initial_dense_units,
            activation=self.classif_activation,
            name=f'dense_0_osteo')(temp_layer_osteo)
        temp_layer_kl = Dense(
            self.n_initial_dense_units,
            activation=self.classif_activation,
            name=f'dense_0_kl')(temp_layer_kl)
        temp_layer_hsa = Dense(
            self.n_initial_dense_units,
            activation=self.classif_activation,
            name=f'dense_0_hsa')(temp_layer_hsa)

        # Dense Layers
        temp_layer_osteo = Dense(
            self.n_initial_dense_units / 8,
            activation=self.classif_activation,
            name=f'dense_1_osteo')(temp_layer_osteo)
        temp_layer_kl = Dense(
            self.n_initial_dense_units / 8,
            activation=self.classif_activation,
            name=f'dense_1_kl')(temp_layer_kl)
        temp_layer_hsa = Dense(
            self.n_initial_dense_units / 8,
            activation=self.classif_activation,
            name=f'dense_1_hsa')(temp_layer_hsa)

        # OS output
        out_osteo = Dense(
            3,
            activation='softmax',
            name=f'out_osteophyte')(temp_layer_osteo)

        # JS output
        out_kl = Dense(
            3,
            activation='softmax',
            name=f'out_impingement')(temp_layer_kl)

        # HSA output
        out_hsa = Dense(
            1,
            activation='sigmoid',
            name=f'out_hsa')(temp_layer_hsa)

        # Define model
        self.model = Model(
            inputs=[input_layer],
            outputs=[out_osteo,
                     out_kl,
                     out_hsa
                     ]
        )

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def plot_model(self):
        '''
        Plot Model
        '''
        tf.keras.utils.plot_model(
            self.model, to_file=os.path.join(os.getcwd(), 'model.png'))

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())

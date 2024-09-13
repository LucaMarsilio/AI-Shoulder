"""
Utils functions.
"""

from scipy.ndimage import distance_transform_edt as distance
from cv2 import findContours
from cv2 import RETR_EXTERNAL, CHAIN_APPROX_NONE, cvtColor, COLOR_BGR2GRAY
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D, GlobalAveragePooling3D, MaxPool3D
from tensorflow.keras.layers import Input, Concatenate, Lambda, Dropout, Concatenate, Multiply, Softmax, Reshape, UpSampling3D, \
    Subtract, Add, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


def calc_DM(seg):
    """
    Computes NON-SIGNED Distance Map of input ground truth image or volume using scipy function.
    In case seg is 3D volume, it separately computes 2D DM fo each single slice.
    Args:
        seg: 2D or 3D binary array to compute the distance map
    Returns:
        res: distance map
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(bool)

    if len(seg.shape) == 2:
        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask + \
                (distance(posmask) - 1) * posmask
        return res
    elif len(seg.shape) == 3:
        for i in range(seg.shape[2]):
            pos = posmask[:, :, i]
            if pos.any():
                neg = ~pos
                res[:, :, i] = distance(neg) * neg + (distance(pos) - 1) * pos
        return res
    else:
        print("Could not recognise dimensions")


def calc_SDM(seg):
    """
    Computes SIGNED Distance Map of input ground truth image or volume using scipy function.
    In case seg is 3D volume, it separately computes 2D SDM fo each single slice.
    Args:
        seg: 2D or 3D binary array to compute the distance map
    Returns:
        res: distance map
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(bool)

    if len(seg.shape) == 2:
        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - \
                (distance(posmask) - 1) * posmask
        return res
    elif len(seg.shape) == 3:
        for i in range(seg.shape[2]):
            pos = posmask[:, :, i]
            if pos.any():
                neg = ~pos
                res[:, :, i] = distance(neg) * neg - (distance(pos) - 1) * pos
        return res
    else:
        print("Could not recognise dimensions")


def calc_DM_edge(seg):
    """
    Computes Non-Signed (Euclidean) Distance Map of input ground-truth volume CONTOURS using scipy function.
    It separately computes 2D Distance Maps for each single slice in the volume.
    Args:
        seg: 3D binary volume of ground truth contours
    Returns:
        res: distance map
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(bool)

    for i in range(seg.shape[2]):
        pos = posmask[:, :, i]
        if pos.any():
            neg = ~pos
            res[:, :, i] = distance(neg)
    return res


def calc_DM_batch(y_true, numClasses):
    """
    Prepares the input for NON-SIGNED Distance Map computation, and pass it to calc_DM
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            dist_batch[c, i] = calc_DM(y)
    return np.array(dist_batch).astype(np.float32)


def calc_SDM_batch(y_true, numClasses):
    """
    Prepares the input for SIGNED Distance Map computation, and pass it to calc_SDM
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    dist_batch = np.zeros_like(y_true_numpy)
    for c in range(numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            dist_batch[c, i] = calc_SDM(y)
    return np.array(dist_batch).astype(np.float32)


def calc_DM_batch_edge(y_true, numClasses):
    """
    Receives y_true mask labels, returns y_true contours and euclidean transform of y_true mask labels. Euclidean
    transform is computed by the function 'calc_DM_edge'.
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of distance map of the same dimension of input tensor
        array of ground truth contours of the same dimension of input tensor
    """
    y_true_numpy = y_true.numpy()
    surface_label = np.zeros_like(y_true_numpy)
    dist_batch = np.zeros_like(y_true_numpy)

    # compute contours for each slice, in each class volume, in each batch.
    for c in range(1, numClasses):  # for each class
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):  # for each batch
            for k in range(y.shape[2]):  # for each slice
                img_lab = y[:, :, k].astype(np.uint8)
                contour_lab, hierarchy_lab = findContours(
                    img_lab, RETR_EXTERNAL, CHAIN_APPROX_NONE)

                if len(contour_lab) != 0:  # if contour per slice is present
                    for j in range(len(contour_lab)):
                        if contour_lab[j].shape[1] == 1:
                            contour_lab[j].resize(contour_lab[j].shape[0], 2)
                        surface_label[c, i, contour_lab[j]
                                      [:, 1], contour_lab[j][:, 0], k] = 1
                else:
                    surface_label[c, i, :, :, k] = np.zeros_like(img_lab)
            # compute Euclidean transform
            dist_batch[c, i] = calc_DM_edge(surface_label[c, i])

        surface_label[0] += surface_label[c]
    for i in range(y_true_numpy.shape[1]):
        dist_batch[0, i] = calc_DM_edge(surface_label[0, i])
    surface_label[0] = 1 - surface_label[0]  # invert background label
    return np.array(dist_batch).astype(
        np.float32), np.array(surface_label).astype(
        np.float32)


def computeContours(y_true, numClasses):
    """
    Receive y_true masks and creates y_true contours.
    Args:
        y_true: ground truth tensor [class, batch, rows, columns, slices] or [class, batch, rows, columns]
        numClasses: number of classes
    Returns:
        array of ground truth contours [class, batch, rows, columns, slices]
        number of contour voxels
    """
    y_true_numpy = y_true.numpy()
    contour_voxels = np.zeros(1)
    surface_label = np.zeros_like(y_true_numpy)
    for c in range(1, numClasses):
        temp_y = y_true_numpy[c]
        for i, y in enumerate(temp_y):
            for k in range(y.shape[2]):
                img_lab = y[:, :, k].astype(np.uint8)
                contour_lab, hierarchy_lab = findContours(
                    img_lab, RETR_EXTERNAL, CHAIN_APPROX_NONE)

                if len(contour_lab) != 0:  # CONTOUR PER SLICE IS PRESENT
                    for j in range(len(contour_lab)):
                        if contour_lab[j].shape[1] == 1:
                            contour_lab[j].resize(contour_lab[j].shape[0], 2)
                        surface_label[c, i, contour_lab[j]
                                      [:, 1], contour_lab[j][:, 0], k] = 1
                else:
                    surface_label[c, i, :, :, k] = np.zeros_like(img_lab)

        surface_label[0] += surface_label[c]
        contour_voxels += np.count_nonzero(surface_label[c])

    return np.array(surface_label).astype(
        np.float32), np.array(contour_voxels).astype(
        np.float32)


def count_class_voxels(labels, nVoxels, numClasses):
    """
    Counts total number of voxels for each class in the batch size.
    input is supposed to be 4 or 5-dimensional: (class, batch, rows, columns) or
    (class, batch, rows, columns, slices).
    Args:
        labels: ground truth tensor of dimensions (class, batch_size, rows, columns, slices) or
        (class, batch_size, rows, columns)
        nVoxels: total number of voxels
        numClasses: number of classes
    Returns:
        out: list with number of voxel per class
    """
    out = [0] * numClasses
    out[0] = 0
    for c in range(1, numClasses):
        out[c] = tf.math.count_nonzero(labels[c])
    first_term = tf.cast(nVoxels, 'int64')
    second_term = tf.reduce_sum(out)
    out[0] = tf.subtract(first_term, second_term)
    return out


def get_loss_weights(labels, nVoxels, numClasses):
    """
    Compute loss weights for each class.
    Args:
        labels: ground truth tensor of dimensions (class, batch_size, rows, columns, slices) or
        (class, batch_size, rows, columns)
        nVoxels: total number of voxels
        numClasses: number of classes
    Returns:
        1D tf.tensor of len = numClasses containing weights for each class
    """

    numerator_1 = count_class_voxels(labels, nVoxels, numClasses)
    numerator_1 = tf.math.sqrt(tf.math.divide_no_nan(1.0, numerator_1))
    numerator_1 = tf.math.divide_no_nan(
        numerator_1, tf.reduce_sum(numerator_1))

    return numerator_1


# Functions for Unet2
def PEE(x, filters, input_dims=3):
    if (input_dims == 2):
        conv_layer = Conv2D
        average_layer = AveragePooling2D
        if filters > 30:
            pool_size_1 = (3, 3)
            pool_size_2 = (5, 5)
            strides = (1, 1)
        else:
            pool_size_1 = (5, 5)
            pool_size_2 = (7, 7)
            strides = (1, 1)
    else:
        conv_layer = Conv3D
        average_layer = AveragePooling3D
        if filters > 30:
            pool_size_1 = (3, 3, 3)
            pool_size_2 = (5, 5, 5)
            strides = (1, 1, 1)
        else:
            pool_size_1 = (5, 5, 5)
            pool_size_2 = (7, 7, 7)
            strides = (1, 1, 1)

    x = conv_layer(filters / 2, strides, padding='same')(x)
    x_1 = average_layer(pool_size=pool_size_1,
                        strides=strides, padding='same')(x)
    x_2 = average_layer(pool_size=pool_size_2,
                        strides=strides, padding='same')(x)

    x_11 = Subtract()([x, x_1])
    x_22 = Subtract()([x, x_2])

    x = Concatenate()([x, x_11, x_22])
    x = conv_layer(filters, strides, padding='same')(x)
    return x


def RA(upsampled, high_level, filters):
    x = Activation('sigmoid')(upsampled)
    x = -1 * x + 1
    x = Multiply()([x, high_level])

    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = Add()([x, upsampled])
    return x


def MINI_MTL(inputs, filters, numClasses, i):
    x_edge = RA(inputs, inputs, filters)
    x_mask = RA(inputs, inputs, filters)

    x_edge = Conv3D(filters, (3, 3, 3), padding='same')(x_edge)
    x_edge = BatchNormalization(axis=-1)(x_edge)
    x_edge = Activation('relu')(x_edge)
    x_mask = Conv3D(filters, (3, 3, 3), padding='same')(x_mask)
    x_mask = BatchNormalization(axis=-1)(x_mask)
    x_mask = Activation('relu')(x_mask)

    out_edge = Conv3D(numClasses - 1, (1, 1, 1), padding='same')(x_edge)
    out_edge = UpSampling3D(pow(2, i))(out_edge)
    out_edge = Softmax(axis=-1, dtype='float32',
                       name='out_edge_{}'.format(i))(out_edge)
    out_mask = Conv3D(numClasses, (1, 1, 1), padding='same')(x_mask)
    out_mask = UpSampling3D(pow(2, i))(out_mask)
    out_mask = Softmax(axis=-1, dtype='float32',
                       name='out_mask_{}'.format(i))(out_mask)

    # out_mtl = Add()([x_mask, x_edge])
    out_mtl = Concatenate()([x_mask, x_edge])
    out_mtl = Conv3D(filters, (1, 1, 1), padding='same')(out_mtl)

    return out_mtl, out_edge, out_mask


def build_MINI_MTL(input_shape, filters, numClasses, i):
    input_layer = Input(shape=(input_shape, input_shape, input_shape, filters))
    x_edge = RA(input_layer, input_layer, filters)
    x_mask = RA(input_layer, input_layer, filters)

    x_edge = Conv3D(filters, (3, 3, 3), padding='same')(x_edge)
    x_edge = BatchNormalization(axis=-1)(x_edge)
    x_edge = Activation('relu')(x_edge)
    x_mask = Conv3D(filters, (3, 3, 3), padding='same')(x_mask)
    x_mask = BatchNormalization(axis=-1)(x_mask)
    x_mask = Activation('relu')(x_mask)

    out_edge = Conv3D(numClasses, (1, 1, 1), padding='same')(x_edge)
    out_edge = Softmax(axis=-1)(out_edge)
    out_edge = UpSampling3D(pow(2, i), name='out_edge_{}'.format(i))(out_edge)
    out_mask = Conv3D(numClasses, (1, 1, 1), padding='same')(x_mask)
    out_mask = Softmax(axis=-1)(out_mask)
    out_mask = UpSampling3D(pow(2, i), name='out_mask_{}'.format(i))(out_mask)

    out_mtl = Concatenate()([x_mask, x_edge])
    out_mtl = Conv3D(filters, (1, 1, 1), padding='same')(out_mtl)

    mtl_model = Model(inputs=[input_layer], outputs=[out_edge, out_mask])

    return mtl_model, out_mtl


def CFF(input_list, input_size, filters, i):
    out_shape = input_size / pow(2, i)

    y = tf.zeros_like(input_list[i - 1])
    for j, x in enumerate(input_list):
        if j < i - 1:
            down_factor = int((input_size / pow(2, j + 1)) / out_shape)
            x = AveragePooling3D((down_factor, down_factor, down_factor))(x)
            x = Conv3D(filters, (1, 1, 1), padding='same')(x)
            sigm = Activation('sigmoid')(x)
            x = Multiply()([x, sigm])
            y = Add()([y, x])
        if j > i - 1:
            up_factor = int(out_shape / (input_size / pow(2, j + 1)))
            x = Conv3D(filters, (1, 1, 1), padding='same')(x)
            x = UpSampling3D((up_factor, up_factor, up_factor))(x)
            sigm = Activation('sigmoid')(x)
            x = Multiply()([x, sigm])
            y = Add()([y, x])

    x_i = input_list[i - 1]
    x_i_sigm = Activation('sigmoid')(x_i)
    x_i_sigm = -1 * x_i_sigm + 1
    out = Multiply()([x_i_sigm, y])
    out = Add()([out, x_i])
    return out


def ASPP(x, filters):
    shape = x.shape

    y1 = AveragePooling3D(pool_size=(shape[1], shape[2], shape[3]))(x)
    y1 = Conv3D(filters / 2, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling3D((shape[1], shape[2], shape[3]))(y1)

    y2 = Conv3D(filters / 2, 1, dilation_rate=1,
                padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv3D(filters / 2, 3, dilation_rate=2,
                padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv3D(filters / 2, 3, dilation_rate=4,
                padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv3D(filters / 2, 3, dilation_rate=8,
                padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv3D(filters, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

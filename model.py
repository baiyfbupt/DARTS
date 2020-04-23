from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import UniformInitializer, ConstantInitializer
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *


def conv_bn(x, c_out, kernel_size, padding, stride, name):
    k = (1. / x.shape[1] / kernel_size / kernel_size)**0.5
    conv1 = fluid.layers.conv2d(
        x,
        c_out,
        kernel_size,
        stride=stride,
        padding=padding,
        param_attr=fluid.ParamAttr(
            name=name + "_conv", initializer=UniformInitializer(
                low=-k, high=k)),
        bias_attr=False)
    bn1 = fluid.layers.batch_norm(
        conv1,
        param_attr=fluid.ParamAttr(
            name=name + "_bn_scale", initializer=ConstantInitializer(value=1)),
        bias_attr=fluid.ParamAttr(
            name=name + "_bn_offset", initializer=ConstantInitializer(value=0)),
        moving_mean_name=name + "_bn_mean",
        moving_variance_name=name + "_bn_variance")
    return bn1


def classifier(x, num_classes, name):
    out = fluid.layers.pool2d(x, pool_type='avg', global_pooling=True)
    out = fluid.layers.squeeze(out, axes=[2, 3])
    k = (1. / out.shape[1])**0.5
    out = fluid.layers.fc(out,
                          num_classes,
                          param_attr=fluid.ParamAttr(
                              name=name + "_fc_weights",
                              initializer=UniformInitializer(
                                  low=-k, high=k)),
                          bias_attr=fluid.ParamAttr(
                              name=name + "_fc_bias",
                              initializer=UniformInitializer(
                                  low=-k, high=k)))
    return out


def drop_path(x, drop_prob, mask):
    keep_prob = 1 - drop_prob[0]
    x = fluid.layers.elementwise_mul(x / keep_prob, mask, axis=0)
    return x


def cell(s0, s1, is_train, genotype, c_curr, reduction, reduction_prev,
         do_drop_path, drop_prob, drop_path_cell, name):
    if reduction:
        op_names, indices = zip(*genotype.reduce)
        concat = genotype.reduce_concat
    else:
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
    num_cells = len(op_names) // 2
    multiplier = len(concat)

    if reduction_prev:
        s0 = factorized_reduce(s0, c_curr, name=name + '_s-2')
    else:
        s0 = relu_conv_bn(s0, c_curr, 1, 1, 0, name=name + '_s-2')
    s1 = relu_conv_bn(s1, c_curr, 1, 1, 0, name=name + '_s-1')

    state = [s0, s1]
    for i in range(num_cells):
        stride = 2 if reduction and indices[2 * i] < 2 else 1
        h1 = OPS[op_names[2 * i]](state[indices[2 * i]], c_curr, stride, True,
                                  name + "_s" + str(i) + "_h1")
        stride = 2 if reduction and indices[2 * i + 1] < 2 else 1
        h2 = OPS[op_names[2 * i + 1]](state[indices[2 * i + 1]], c_curr, stride,
                                      True, name + "_s" + str(i) + "_h2")
        if is_train and do_drop_path:
            if op_names[2 * i] is not 'skip_connect':
                h1 = drop_path(h1, drop_prob, drop_path_cell[:, i, 0])
            if op_names[2 * i + 1] is not 'skip_connect':
                h2 = drop_path(h2, drop_prob, drop_path_cell[:, i, 1])
        state.append(h1 + h2)
    out = fluid.layers.concat(input=state[-multiplier:], axis=1)
    return out


def auxiliary_cifar(x, num_classes, name):
    x = fluid.layers.relu(x)
    pooled = fluid.layers.pool2d(
        x, pool_size=5, pool_stride=3, pool_padding=0, pool_type='avg')
    conv1 = conv_bn(
        x=pooled,
        c_out=128,
        kernel_size=1,
        padding=0,
        stride=1,
        name=name + '_conv_bn1')
    conv1 = fluid.layers.relu(conv1)
    conv2 = conv_bn(
        x=conv1,
        c_out=768,
        kernel_size=2,
        padding=0,
        stride=1,
        name=name + '_conv_bn2')
    conv2 = fluid.layers.relu(conv2)
    out = classifier(conv2, num_classes, name)
    return out


def network_cifar(x, is_train, c_in, num_classes, layers, auxiliary, genotype,
                  do_drop_path, drop_prob, drop_path_mask, name):
    stem_multiplier = 3
    c_curr = stem_multiplier * c_in
    x = conv_bn(
        x=x,
        c_out=c_curr,
        kernel_size=3,
        padding=1,
        stride=1,
        name=name + '_s0_conv_bn')
    s0 = s1 = x
    reduction_prev = False
    logits_aux = None
    c_curr = c_in
    for i in range(layers):
        if i in [layers // 3, 2 * layers // 3]:
            c_curr *= 2
            reduction = True
        else:
            reduction = False
        if do_drop_path and is_train:
            drop_path_cell = drop_path_mask[:, i, :, :]
        else:
            drop_path_cell = drop_path_mask
        s0, s1 = s1, cell(s0, s1, is_train, genotype, c_curr, reduction,
                          reduction_prev, do_drop_path, drop_prob,
                          drop_path_cell, name + "_l" + str(i))
        reduction_prev = reduction
        if i == 2 * layers // 3:
            if auxiliary and is_train:
                logits_aux = auxiliary_cifar(s1, num_classes,
                                             name + "_l" + str(i) + "_aux")

    logits = classifier(s1, num_classes, name)
    return logits, logits_aux


def auxiliary_imagenet(x, num_classes, name):
    x = fluid.layers.relu(x)
    pooled = fluid.layers.pool2d(
        x, pool_size=5, pool_stride=2, pool_padding=0, pool_type='avg')
    conv1 = conv_bn(
        x=pooled,
        c_out=128,
        kernel_size=1,
        padding=0,
        stride=1,
        name=name + '_conv_bn1')
    conv1 = fluid.layers.relu(conv1)
    conv2 = conv_bn(
        x=conv1,
        c_out=768,
        kernel_size=2,
        padding=0,
        stride=1,
        name=name + '_conv_bn2')
    conv2 = fluid.layers.relu(conv2)
    out = classifier(conv2, num_classes, name)
    return out


def network_imagenet(x, is_train, c_in, num_classes, layers, auxiliary,
                     genotype, name):
    x = conv_bn(
        x=x,
        c_out=c_in // 2,
        kernel_size=3,
        padding=1,
        stride=2,
        name=name + '_conv_bn_s0_0')
    x = fluid.layers.relu(x)
    s0 = conv_bn(
        x=x,
        c_out=c_in,
        kernel_size=3,
        padding=1,
        stride=2,
        name=name + '_conv_bn_s0_1')
    s1 = fluid.layers.relu(s0)
    s1 = conv_bn(
        x=s1,
        c_out=c_in,
        kernel_size=3,
        padding=1,
        stride=2,
        name=name + '_conv_bn_s1')
    reduction_prev = True
    logits_aux = None
    c_curr = c_in
    for i in range(layers):
        if i in [layers // 3, 2 * layers // 3]:
            c_curr *= 2
            reduction = True
        else:
            reduction = False
        s0, s1 = s1, cell(s0, s1, is_train, genotype, c_curr, reduction,
                          reduction_prev, False, '', '', name + "_l" + str(i))
        reduction_prev = reduction
        if i == 2 * layers // 3:
            if auxiliary and is_train:
                logits_aux = auxiliary_imagenet(s1, num_classes,
                                                name + "_l" + str(i) + "_aux")

    logits = classifier(s1, num_classes, name)
    return logits, logits_aux

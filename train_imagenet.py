from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
import shutil
import argparse
import functools
import numpy as np

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

import paddle
import paddle.fluid as fluid
from model import network_imagenet as network
import genotypes
import reader
import utility

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('num_workers',       int,   4,               "The multiprocess reader number.")
add_arg('data_dir',          str,   'dataset/ILSVRC2012',"The dir of dataset.")
add_arg('batch_size',        int,   128,             "Minibatch size.")
add_arg('learning_rate',     float, 0.1,             "The start learning rate.")
add_arg('decay_rate',        float, 0.97,            "The lr decay rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('weight_decay',      float, 3e-5,            "Weight_decay.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   250,             "Epoch number.")
add_arg('init_channels',     int,   48,              "Init channel number.")
add_arg('layers',            int,   14,              "Total number of layers.")
add_arg('class_num',         int,   1000,            "Class number of dataset.")
add_arg('trainset_num',      int,   1281167,         "Images number of trainset.")
add_arg('model_save_dir',    str,   'eval_imagenet', "The path to save model.")
add_arg('auxiliary',         bool,  True,            'Use auxiliary tower.')
add_arg('auxiliary_weight',  float, 0.4,             "Weight for auxiliary loss.")
add_arg('drop_path_prob',    float, 0.0,             "Drop path probability.")
add_arg('dropout',           float, 0.0,             "Dropout probability.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('image_shape',       str,   '3,224,224',     "Input image size")
add_arg('label_smooth',      float, 0.1,             "Label smoothing.")
add_arg('arch',              str,   'DARTS_PADDLE',  "Which architecture to use")
add_arg('report_freq',       int,   100,             'Report frequency')
add_arg('with_mem_opt',      bool,  True,            "Whether to use memory optimization or not.")
# yapf: enable


def cross_entropy_label_smooth(preds, targets, epsilon):
    preds = fluid.layers.softmax(preds)
    targets_one_hot = fluid.layers.one_hot(input=targets, depth=args.class_num)
    targets_smooth = fluid.layers.label_smooth(
        targets_one_hot, epsilon=epsilon, dtype="float32")
    loss = fluid.layers.cross_entropy(
        input=preds, label=targets_smooth, soft_label=True)
    return loss


def build_program(main_prog, startup_prog, is_train, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.data(
                name="image", shape=[None] + image_shape, dtype="float32")
            label = fluid.data(name="label", shape=[None, 1], dtype="int64")
            data_loader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label],
                capacity=64,
                use_double_buffer=True,
                iterable=True)
            genotype = eval("genotypes.%s" % args.arch)
            logits, logits_aux = network(
                x=image,
                is_train=is_train,
                c_in=args.init_channels,
                num_classes=args.class_num,
                layers=args.layers,
                auxiliary=args.auxiliary,
                genotype=genotype,
                name='model')
            top1 = fluid.layers.accuracy(input=logits, label=label, k=1)
            top5 = fluid.layers.accuracy(input=logits, label=label, k=5)
            loss = fluid.layers.reduce_mean(
                cross_entropy_label_smooth(logits, label, args.label_smooth))
            if is_train:
                if args.auxiliary:
                    loss_aux = fluid.layers.reduce_mean(
                        cross_entropy_label_smooth(logits_aux, label,
                                                   args.label_smooth))
                    loss = loss + args.auxiliary_weight * loss_aux
                step_per_epoch = int(args.trainset_num / args.batch_size)
                learning_rate = fluid.layers.exponential_decay(
                    args.learning_rate,
                    step_per_epoch,
                    args.decay_rate,
                    staircase=True)
                fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))
                optimizer = fluid.optimizer.MomentumOptimizer(
                    learning_rate,
                    args.momentum,
                    regularization=fluid.regularizer.L2DecayRegularizer(
                        args.weight_decay))
                optimizer.minimize(loss)
                outs = [loss, top1, top5, learning_rate]
            else:
                outs = [loss, top1, top5]
    return outs, data_loader


def train(main_prog, exe, epoch_id, train_loader, fetch_list, args):
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for step_id, data in enumerate(train_loader()):
        loss_v, top1_v, top5_v, lr = exe.run(
            main_prog, feed=data, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % args.report_freq == 0:
            logger.info(
                "Train Epoch {}, Step {}, Lr {:.8f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch_id, step_id, lr[0], loss.avg[0], top1.avg[0],
                       top5.avg[0]))
    return top1.avg[0], top5.avg[0]


def valid(main_prog, exe, epoch_id, valid_loader, fetch_list, args):
    loss = utility.AvgrageMeter()
    top1 = utility.AvgrageMeter()
    top5 = utility.AvgrageMeter()
    for step_id, data in enumerate(valid_loader()):
        loss_v, top1_v, top5_v = exe.run(
            main_prog, feed=data, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % args.report_freq == 0:
            logger.info(
                "Valid Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch_id, step_id, loss.avg[0], top1.avg[0], top5.avg[
                    0]))
    return top1.avg[0], top5.avg[0]


def main(args):
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    valid_prog = fluid.Program()

    train_fetch_list, train_loader = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        is_train=True,
        args=args)
    valid_fetch_list, valid_loader = build_program(
        main_prog=valid_prog,
        startup_prog=startup_prog,
        is_train=False,
        args=args)

    logger.info("param size = {:.6f}MB".format(
        utility.count_parameters_in_MB(train_prog.global_block()
                                       .all_parameters(), 'model')))
    valid_prog = valid_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    train_reader = paddle.batch(
        reader.imagenet_reader(args.data_dir, 'train'),
        batch_size=args.batch_size,
        drop_last=True)
    valid_reader = paddle.batch(
        reader.imagenet_reader(args.data_dir, 'val'),
        batch_size=args.batch_size)

    places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    train_loader.set_sample_list_generator(train_reader, places)
    valid_loader.set_sample_list_generator(valid_reader, place)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4 * devices_num
    build_strategy = fluid.BuildStrategy()
    if args.with_mem_opt:
        train_fetch_list[0].persistable = True
        train_fetch_list[1].persistable = True
        train_fetch_list[2].persistable = True
        train_fetch_list[3].persistable = True
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True

    parallel_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=train_fetch_list[0].name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    valid_prog = fluid.CompiledProgram(valid_prog)

    def save_model(postfix, program):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        logger.info('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    best_valid_top1 = 0
    for epoch_id in range(args.epochs):
        train_top1, train_top5 = train(parallel_train_prog, exe, epoch_id,
                                       train_loader, train_fetch_list, args)
        logger.info("Epoch {}, train_top1 {:.6f}, train_top5 {:.6f}".format(
            epoch_id, train_top1, train_top5))
        valid_top1, valid_top5 = valid(valid_prog, exe, epoch_id, valid_loader,
                                       valid_fetch_list, args)
        if valid_top1 > best_valid_top1:
            best_valid_top1 = valid_top1
            save_model('imagenet_model', train_prog)
        logger.info(
            "Epoch {}, valid_top1 {:.6f}, valid_top5 {:.6f}, best_valid_top1 {:6f}".
            format(epoch_id, valid_top1, valid_top5, best_valid_top1))


if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)

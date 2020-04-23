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

import paddle.fluid as fluid
from model import network_cifar as network
import genotypes
import reader
import utility

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utility.add_arguments, argparser=parser)

# yapf: disable
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('num_workers',       int,   4,               "The multiprocess reader number.")
add_arg('data',              str,   'dataset/cifar10',"The dir of dataset.")
add_arg('batch_size',        int,   96,              "Minibatch size.")
add_arg('learning_rate',     float, 0.025,           "The start learning rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('weight_decay',      float, 3e-4,            "Weight_decay.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   600,             "Epoch number.")
add_arg('init_channels',     int,   36,              "Init channel number.")
add_arg('layers',            int,   20,              "Total number of layers.")
add_arg('class_num',         int,   10,              "Class number of dataset.")
add_arg('trainset_num',      int,   50000,           "images number of trainset.")
add_arg('model_save_dir',    str,   'eval_cifar10',   "The path to save model.")
add_arg('cutout',            bool,  True,            'Whether use cutout.')
add_arg('cutout_length',     int,   16,              "Cutout length.")
add_arg('auxiliary',         bool,  True,            'Use auxiliary tower.')
add_arg('auxiliary_weight',  float, 0.4,             "Weight for auxiliary loss.")
add_arg('drop_path_prob',    float, 0.2,             "Drop path probability.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('image_shape',       str,   "3,32,32",       "Input image size")
add_arg('arch',              str,   'DARTS_PADDLE',  "Which architecture to use")
add_arg('report_freq',       int,   50,              'Report frequency')
add_arg('with_mem_opt',      bool,  True,            "Whether to use memory optimization or not.")
# yapf: enable


def build_program(main_prog, startup_prog, is_train, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    num_cells = 4
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
            drop_path_prob = ''
            drop_path_mask = ''
            if args.drop_path_prob > 0 and is_train:
                drop_path_prob = fluid.data(
                    name="drop_path_prob",
                    shape=[args.batch_size, 1],
                    dtype="float32")
                drop_path_mask = fluid.data(
                    name="drop_path_mask",
                    shape=[args.batch_size, args.layers, num_cells, 2],
                    dtype="float32")
            genotype = eval("genotypes.%s" % args.arch)
            do_drop_path = args.drop_path_prob > 0
            logits, logits_aux = network(
                x=image,
                is_train=is_train,
                c_in=args.init_channels,
                num_classes=args.class_num,
                layers=args.layers,
                auxiliary=args.auxiliary,
                genotype=genotype,
                do_drop_path=do_drop_path,
                drop_prob=drop_path_prob,
                drop_path_mask=drop_path_mask,
                name='model')
            top1 = fluid.layers.accuracy(input=logits, label=label, k=1)
            top5 = fluid.layers.accuracy(input=logits, label=label, k=5)
            loss = fluid.layers.reduce_mean(
                fluid.layers.softmax_with_cross_entropy(logits, label))
            if is_train:
                if args.auxiliary:
                    loss_aux = fluid.layers.reduce_mean(
                        fluid.layers.softmax_with_cross_entropy(logits_aux,
                                                                label))
                    loss = loss + args.auxiliary_weight * loss_aux
                step_per_epoch = int(args.trainset_num / args.batch_size)
                learning_rate = fluid.layers.cosine_decay(
                    args.learning_rate, step_per_epoch, args.epochs)

                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=args.grad_clip)
                optimizer = fluid.optimizer.MomentumOptimizer(
                    learning_rate,
                    args.momentum,
                    regularization=fluid.regularizer.L2DecayRegularizer(
                        args.weight_decay),
                    grad_clip=clip)
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
        devices_num = len(data)
        if args.drop_path_prob > 0:
            feed = []
            for device_id in range(devices_num):
                image = data[device_id]['image']
                label = data[device_id]['label']
                num_cells = 4
                drop_path_prob = np.array(
                    [[args.drop_path_prob * epoch_id / args.epochs]
                     for i in range(args.batch_size)]).astype(np.float32)
                drop_path_mask = 1 - np.random.binomial(
                    1,
                    drop_path_prob[0],
                    size=[args.batch_size, args.layers, num_cells, 2]).astype(
                        np.float32)
                feed.append({
                    "image": image,
                    "label": label,
                    "drop_path_prob": drop_path_prob,
                    "drop_path_mask": drop_path_mask
                })
        else:
            feed = data
        loss_v, top1_v, top5_v, lr = exe.run(
            main_prog, feed=feed, fetch_list=[v.name for v in fetch_list])
        loss.update(loss_v, args.batch_size)
        top1.update(top1_v, args.batch_size)
        top5.update(top5_v, args.batch_size)
        if step_id % args.report_freq == 0:
            logger.info(
                "Train Epoch {}, Step {}, Lr {:.8f}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch_id, step_id, lr[0], loss.avg[0], top1.avg[0],
                       top5.avg[0]))
    return top1.avg[0]


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
    return top1.avg[0]


def main(args):
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    is_shuffle = True

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    train_fetch_list, train_loader = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        is_train=True,
        args=args)
    valid_fetch_list, valid_loader = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        is_train=False,
        args=args)

    logger.info("param size = {:.6f}MB".format(
        utility.count_parameters_in_MB(train_prog.global_block()
                                       .all_parameters(), 'model')))
    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    train_reader = reader.train_valid(
        batch_size=args.batch_size,
        is_train=True,
        is_shuffle=is_shuffle,
        args=args)
    valid_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=False, is_shuffle=False, args=args)

    places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    train_loader.set_batch_generator(train_reader, places=places)
    valid_loader.set_batch_generator(valid_reader, places=place)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 4 * devices_num
    build_strategy = fluid.BuildStrategy()
    if args.with_mem_opt:
        for i in range(len(train_fetch_list)):
            train_fetch_list[i].persistable = True
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True

    parallel_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=train_fetch_list[0].name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    test_prog = fluid.CompiledProgram(test_prog)

    def save_model(postfix, program):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        logger.info('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=program)

    best_acc = 0
    for epoch_id in range(args.epochs):
        train_top1 = train(parallel_train_prog, exe, epoch_id, train_loader,
                           train_fetch_list, args)
        logger.info("Epoch {}, train_acc {:.6f}".format(epoch_id, train_top1))
        valid_top1 = valid(test_prog, exe, epoch_id, valid_loader,
                           valid_fetch_list, args)
        if valid_top1 > best_acc:
            best_acc = valid_top1
            save_model('cifar10_model', train_prog)
        logger.info("Epoch {}, valid_acc {:.6f}, best_valid_acc {:.6f}".format(
            epoch_id, valid_top1, best_acc))


if __name__ == '__main__':
    args = parser.parse_args()
    utility.print_arguments(args)
    utility.check_cuda(args.use_gpu)

    main(args)

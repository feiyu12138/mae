# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.247, 0.243, 0.261)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_dataset_cifar10(is_train, args):
    transform = build_transform_cifar10(is_train, args)

    dataset = datasets.CIFAR10(root=args.data_path, train=is_train, transform=transform, download=True)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_transform_cifar10(is_train, args):
    mean = CIFAR10_DEFAULT_MEAN
    std = CIFAR10_DEFAULT_STD
    # train transform
    # if is_train:
    #     # this should always dispatch to transforms_imagenet_train
    #     transform = create_transform(
    #         input_size=args.input_size,
    #         is_training=True,
    #         color_jitter=args.color_jitter,
    #         auto_augment=args.aa,
    #         interpolation='bicubic',
    #         re_prob=args.reprob,
    #         re_mode=args.remode,
    #         re_count=args.recount,
    #         mean=mean,
    #         std=std,
    #     )
    #     return transform

    # eval transform
    t = []
    # if args.input_size <= 32:
    #     crop_pct = 32 / 40 # 0.8
    # else:
    #     crop_pct = 1.0
    # size = int(args.input_size / crop_pct)
    # t.append(
    #     transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    # )
    # t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

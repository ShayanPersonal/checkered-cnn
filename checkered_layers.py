"""
This file implements the building blocks and methods used by checkered CNNs.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from generate import generate_sequence


# 2D checkered CNN layers are implemented by repurposing 3D layers.
class CheckeredConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CheckeredConv2d, self).__init__()
        stride = stride if type(stride) is int else stride[0]
        padding = padding if type(padding) is int else padding[0]
        kernel_size = kernel_size if type(kernel_size) is int else kernel_size[0]
        dilation = dilation if type(dilation) is int else dilation[0]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.depth = depth
        self.conv = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=(depth, kernel_size, kernel_size), stride=(1, stride, stride), 
                                    padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return checkered_forward(self, x)


class CheckeredMaxpool2d(nn.Module):
    def __init__(self, kernel_size, depth=1, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(CheckeredMaxpool2d, self).__init__()
        stride = stride if type(stride) is int else stride[0]
        padding = padding if type(padding) is int else padding[0]
        kernel_size = kernel_size if type(kernel_size) is int else kernel_size[0]
        dilation = dilation if type(dilation) is int else dilation[0]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.depth = depth
        self.conv = nn.MaxPool3d(kernel_size=(depth, kernel_size, kernel_size), stride=(1, stride, stride), 
                        padding=0, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        return checkered_forward(self, x)


class CheckeredAvgPool2d(nn.Module):
    def __init__(self, kernel_size, depth=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True):
        super(CheckeredAvgPool2d, self).__init__()
        stride = stride if type(stride) is int else stride[0]
        padding = padding if type(padding) is int else padding[0]
        kernel_size = kernel_size if type(kernel_size) is int else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.depth = depth
        self.conv = nn.AvgPool3d(kernel_size=(depth, kernel_size, kernel_size), stride=(1, stride, stride), 
                        padding=0, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

    def forward(self, x):
        return checkered_forward(self, x)


class TransposeCheckeredConv2d(nn.Module):
    # Experimental implementation of a deconvolutional layer for checkered submaps (our method is similar to subpixel convolutions)
    # Depth must be equal to depth of the input
    # TODO: Is there a better way to do this?
    def __init__(self, in_channels, out_channels, kernel_size=1, depth=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(TransposeCheckeredConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.input_depth = depth
        self.output_depth = depth // 2
        self.conv = nn.Conv3d(in_channels, out_channels*self.output_depth*4, 
                                    kernel_size=(depth, kernel_size, kernel_size), stride=(1, stride, stride), 
                                    padding=(0, padding, padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x
    
    def shuffle(self, x):
        batch_size, channels, submaps, height, width = x.size()
        out_channels = channels // self.output_depth // 4
        x = x.view(batch_size, out_channels, self.output_depth, 2, 2, height, width)
        x = x.permute(0, 1, 2, 5, 3, 6, 4).contiguous()
        x = x.view(batch_size, out_channels, self.output_depth, height*2, width*2)
        return x


def checkered_forward(self, x):
    # All layers in a CCNN use this function.
    assert self.stride < 3, """You're trying to use stride > 2 and checkered layers are designed for stride=1 or stride=2.
        You'll have to write your own multisampling function if you want longer strides."""

    # Say we want depth > 1 and want the number of outputs submaps to be equal to the number of input submaps.
    # We can't just pad the submap dimension with 0s (it might work in practice but it doesn't make sense). 
    # Instead, we let the the stride continue past the last submap into the next row of the image. This hasn't 
    # been tested very much so its not implemented, but you should be aware that having a custom depth across the submap dimension 
    # can introduce unique problems.
    #if 1 < self.depth < x.size(2):
    #    x = torch.cat((x, F.pad(x[:, :, -self.depth+1:, 1:, :], (0, 0, 0, 1, 0, 0), mode="constant", value=0)), 2)

    if self.padding:
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding, 0, 0), mode="constant", value=0)

    # If stride length is 1, no checkered subsampling is necessary. Simply apply CCNN layer (just a 3D layer with stride=1).
    if self.stride == 1:
        return self.conv(x)

    # Performs complete multisampling with a 2x2 sampler without concatenating the outputs. We're going to throw away half the data 
    # these four lines generate further down below. Note there's an alternative way to do this (run conv once with stride=1 then do all 
    # subsampling yourself) but it seems to perform slightly worse on some experiments due to some bug/quirk with Pytorch.
    y_ul = self.conv(x)
    y_dr = self.conv(F.pad(x[:, :, :, 1:, 1:], (0, 1, 0, 1, 0, 0), mode="constant", value=0))
    y_ur = self.conv(F.pad(x[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0), mode="constant", value=0))
    y_dl = self.conv(F.pad(x[:, :, :, 1:, :], (0, 0, 0, 1, 0, 0), mode="constant", value=0))

    # We use the naive method on our experiments with CIFAR (applying the same sampler on every submap)
    # Consider lattice sampling and random sampling in other tasks, especially when dealing with high-resolution images.
    submap_count = x.size(2)
    sequence = generate_sequence(submap_count, "naive")

    """
    Random samplers during training, lattice samplers at test time.

    Be careful about using 3D convolutions that convolve multiple submaps at a time (depth > 1) if you use random sampling, 
    because randomly applying samplers makes the structure of the submap dimension inconsistent and may make it difficult 
    for 3D kernels to learn.

    This isn't a problem if you process each submap independently (set depth=1 for everything) as in most of our experiments,

    if self.train:
        sequence = generate_sequence(submap_count, "random")
    else:
        sequence = generate_sequence(submap_count, "lattice")
    """

    top_submaps = []
    bottom_submaps = []
    for i, sampler in enumerate(sequence):
        # Apply either sampler 0 (picks top left and bottom right) or sampler 1 (picks top right and bottom left) 
        # onto the submap at index i
        if sampler == 0:
            top_submaps.append(y_ul[:, :, i, :, :])
            bottom_submaps.append(y_dr[:, :, i, :, :])
        else:
            top_submaps.append(y_ur[:, :, i, :, :])
            bottom_submaps.append(y_dl[:, :, i, :, :])

    return torch.stack(top_submaps + bottom_submaps, 2)


def three_over_four_multisampling_forward(self, x):
    # Experimental forward pass that chooses 3/4 samples from 2x2 sampling windows. May improve accuracy further,
    # at the cost of increased computation.
    assert self.stride < 3

    #if 1 < self.depth < x.size(2):
    #    x = torch.cat((x, F.pad(x[:, :, -self.depth+1:, 1:, :], (0, 0, 0, 1, 0, 0), mode="constant", value=0)), 2)

    if self.padding:
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding, 0, 0), mode="constant", value=0)

    # If stride length is 1, no subsampling is necessary. Simply apply conv layer (just a 3D layer with stride=1).
    if self.stride == 1:
        return self.conv(x)

    y_ul = self.conv(x)
    y_dr = self.conv(F.pad(x[:, :, :, 1:, 1:], (0, 1, 0, 1, 0, 0), mode="constant", value=0))
    y_ur = self.conv(F.pad(x[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0), mode="constant", value=0))
    y_dl = self.conv(F.pad(x[:, :, :, 1:, :], (0, 0, 0, 1, 0, 0), mode="constant", value=0))

    submap_count = x.size(2)

    tl_submaps = []
    tr_submaps = []
    br_submaps = []
    for i in range(submap_count):
        tl_submaps.append(y_ul[:, :, i, :, :])
        br_submaps.append(y_dr[:, :, i, :, :])
        tr_submaps.append(y_ur[:, :, i, :, :])

    return torch.stack(tl_submaps + br_submaps + tr_submaps, 2)


def complete_multisampling_forward(self, x):
    # Complete multisampling performs no subsampling. Equivalent to increasing dilation of all proceeding layers.
    # Definitely not recommended due to the extreme performance impact.
    # If you want to experiment with it, replace checkered_forward calls with complete_multisampling_forward.

    # Weird stuff is required when we have depth > 1 and we want to preserve our count of submaps.
    #if 1 < self.depth < x.size(2):
    #    x = torch.cat((x, F.pad(x[:, :, -self.depth+1:, 1:, :], (0, 0, 0, 1, 0, 0), mode="constant", value=0)), 2)

    if self.padding:
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding, 0, 0), mode="constant", value=0)

    # No need for anything fancy if stride=1
    if self.stride == 1:
        return self.conv(x)

    # This works with any stride length. Performs k by k complete multisampling where k is the stride length.
    submap_groups = []
    for i in range(self.stride):
        for j in range(self.stride):
            shifted_x = F.pad(x[:, :, :, j:, i:], (0, i, 0, j, 0, 0), mode="constant", value=0)
            submap_groups.append(self.conv(shifted_x))

    return torch.cat(submap_groups, 2)


def convert_to_checkered(module):
    # This method converts all layers to 3D CCNN layers and transfers over your old parameters.
    # Note that you will likely also have to modify the forward() method of your CNN to handle submaps.
    # Pass this to .apply(). 
    # Example:
    # model = resnet18(pretrained=True)
    # model.apply(convert_to_checkered)
    for name, child in module.named_children():
        classname = child.__class__.__name__
        if classname == 'Conv2d':
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, weight = child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding, child.dilation, child.groups, child.bias, child.weight
            replacement = CheckeredConv2d(in_channels, out_channels, kernel_size, 1, stride, padding, dilation, groups, True if bias is not None else bias)
            replacement.conv.weight, replacement.conv.bias = nn.Parameter(child.weight.unsqueeze(2).data), child.bias
            setattr(module, name, replacement)
        elif classname == 'BatchNorm2d':
            num_features, eps, momentum, affine, weight, bias, running_mean, running_var = child.num_features, child.eps, child.momentum, child.affine, child.weight, child.bias, child.running_mean, child.running_var
            replacement = nn.BatchNorm3d(num_features, eps, momentum, affine)
            replacement.weight, replacement.bias, replacement.running_mean, replacement.running_var = weight, bias, running_mean, running_var
            setattr(module, name, replacement)
        elif classname == 'Dropout2d':
            p, inplace = child.p, child.inplace
            replacement = nn.Dropout3d(p=p, inplace=inplace)
            setattr(module, name, replacement)
        elif classname == 'AdaptiveMaxPool2d':
            output_size, return_indices = child.output_size, child.return_indices
            replacement = nn.AdaptiveMaxPool3d(output_size, return_indices)
            setattr(module, name, replacement)
        elif classname == 'AdaptiveAvgPool2d':
            output_size = child.output_size
            replacement = nn.AdaptiveAvgPool3d(output_size)
            setattr(module, name, replacement)
        elif classname == 'MaxPool2d':
            kernel_size, stride, padding, dilation, return_indices, ceil_mode = child.kernel_size, child.stride, child.padding, child.dilation, child.return_indices, child.ceil_mode
            replacement = CheckeredMaxpool2d(kernel_size, 1, stride, padding, dilation, return_indices, ceil_mode)
            setattr(module, name, replacement)
        elif classname == 'AvgPool2d':
            kernel_size, stride, padding, ceil_mode, count_include_pad = child.kernel_size, child.stride, child.padding, child.ceil_mode, child.count_include_pad
            replacement = CheckeredAvgPool2d(kernel_size, 1, stride, padding, ceil_mode, count_include_pad)
            setattr(module, name, replacement)
        elif classname not in ['SELU', 'Tanh', 'ReLU', 'Sigmoid', 'Sequential', 'Bottleneck', 'Linear']:
            print("Warning: Skipped a child that isn't handled by the conversion script: {}".format(classname))
            print("This is only a problem for low-level layers that expect input with 2 spatial dimensions (e.g. if the layer name ends with '2d').")
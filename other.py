import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from generate import lattice_generator

##################################################################################################
## The following methods were used in undocumented experiments in upsampling / downsampling images 
## with checkered sampling.
##################################################################################################

def shift_map(x, direction):
    if direction == "right":
        return F.pad(x[:, :, :, :, 1:], (1, 0, 0, 0, 0, 0), mode="constant", value=0)
    if direction == "left":
        return F.pad(x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0), mode="constant", value=0)
    if direction == "up":
        return F.pad(x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0), mode="constant", value=0)
    if direction == "down":
        return F.pad(x[:, :, :, 1:, :], (0, 0, 1, 0, 0, 0), mode="constant", value=0)
    raise ValueError

def get_upsample_filter(size):
    factor = (size+1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

def checkered_upsample_bilinear(x, conv=None, size=None):
    x = x.permute(2, 0, 1, 3, 4)
    submap_count = temp = x.size(0)
    subsample_count = 0
    while temp > 1:
        temp >>= 1
        subsample_count += 1
    for i in reversed(range(subsample_count)):
        upsampled_maps = []
        sequence = lattice_generator[i]
        for s, submap1, submap2 in zip(sequence, x[:len(x) // 2], x[len(x) // 2:]):
            upsampled_submap = torch.zeros(submap1.size(0), submap1.size(1), submap1.size(2) * 2, submap1.size(3) * 2).cuda()
            if s == 0:
                upsampled_submap[:, :, ::2, ::2] = submap1
                upsampled_submap[:, :, 1::2, 1::2] = submap2
            else:
                upsampled_submap[:, :, ::2, 1::2] = submap1
                upsampled_submap[:, :, 1::2, ::2] = submap2
            upsampled_maps.append(upsampled_submap)
        x = torch.stack(upsampled_maps, 0)

    if conv:
        # Use the learned upsampler
        x = conv(x.permute(1, 2, 0, 3, 4))
        return x
    else:
        # Use a hard-coded upsampler
        x = torch.squeeze(x, 0)
        kernel_size = max((submap_count // 2) - 1, 3)
        #bilinear_filter = torch.ones(x.size(1), 1, kernel_size, kernel_size).cuda()
        bilinear_filter = get_upsample_filter(kernel_size)
        bilinear_filter = bilinear_filter.view(1, 1, bilinear_filter.size(0), bilinear_filter.size(1)).repeat(x.size(1), 1, 1, 1).cuda()

        x = F.conv2d(x, bilinear_filter, padding=kernel_size // 2, groups=x.size(1))
        #x = F.max_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)

        #if size and (size[0] != x.size(2) or size[1] != x.size(3)):
        #    x = F.upsample_bilinear(x, size=size)

    return x

def checkered_downsample(x, count):
    x = x.unsqueeze(2)
    for c in range(count):
        cut_y, cut_x = x.size(3) % 2, x.size(4) % 2
        y = x[:, :, :, cut_y::2, cut_x::2]
        y_dr = x[:, :, :, 1::2, 1::2]
        y_d = x[:, :, :, cut_y::2, 1::2]
        y_r = x[:, :, :, 1::2, cut_x::2]
        sequence = lattice_generator[c]
        maps1 = []
        maps2 = []
        for i, val in enumerate(sequence):
            if val == 0:
                maps1.append(y[:, :, i, :, :])
                maps2.append(y_dr[:, :, i, :, :])
            else:
                maps1.append(y_d[:, :, i, :, :])
                maps2.append(y_r[:, :, i, :, :])
        x = torch.stack(maps1 + maps2, 2)
    return x

######################################################
# End experiments with upsampling / downsampling images
######################################################

"""
def convert_to_dilated(cnn):
    # Pass CNN directly to this method, do not pass method to .apply(). This method modifies the CNN in-place
    # Works only on 2D networks and square stride lengths/symmetric padding. This version was meant for ResNet
    # This method is still very buggy. It doesn't help that dilated pooling layers in Pytorch are broken.
    # Actually, just consider this broken. Use complete multisampling instead.
    curr_dilation = 1
    for i, child in enumerate(cnn.modules()):
        classname = child.__class__.__name__
        if classname in ('Conv2d', 'MaxPool2d', 'AvgPool2d'):
            if type(child.stride) is tuple:
                child_stride = child.stride[0]
            if type(child.padding) is tuple:
                child_padding = child.padding[0]
            if type(child.padding) is tuple:
                child_dilation = child.dilation[0]
            if type(child.kernel_size) is tuple:
                child_kernel_size = child.kernel_size[0]
            if classname in ('Conv2d'):
                child.dilation = tuple([child_dilation * curr_dilation])
                child.padding = tuple([child_padding * curr_dilation]*2)
                child.stride = tuple([1,1])
                # Warning: This if-statement is a hack for ignoring residual connections in ResNet
                if child_kernel_size > 1:
                    curr_dilation *= child_stride
            if classname in ('MaxPool2d'):
                child.stride = tuple([1,1])
                if child_kernel_size > 1:
                    curr_dilation *= child_stride
"""
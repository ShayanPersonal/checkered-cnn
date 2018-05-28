"""
Demo script for training checkered convolutional neural networks.
This script is based off of the demo script in
https://github.com/gpleiss/efficient_densenet_pytorch
"""

import fire
import os
import time
import torch
import torchvision
from torchvision import datasets, transforms
from models import *

from checkered_layers import convert_to_checkered

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=200, convert=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Add submap dimension if we converted the model to a CCNN.
        if convert:
            input = input.unsqueeze(2)

        if torch.cuda.is_available():
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        input.requires_grad_()

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # print stats
    res = '\t'.join([
        'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
        'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
        'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
        'Loss %.4f (%.4f)' % (losses.val, losses.avg),
        'Error %.4f (%.4f)' % (error.val, error.avg),
    ])
    print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, print_freq=30, is_test=True, convert=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Add submap dimension if we converted the model to a CCNN.
        if convert:
            input = input.unsqueeze(2)

        if torch.cuda.is_available():
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # print stats
    res = '\t'.join([
        'Test' if is_test else 'Valid',
        'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
        'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
        'Loss %.4f (%.4f)' % (losses.val, losses.avg),
        'Error %.4f (%.4f)' % (error.val, error.avg),
    ])
    print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, test_set, save, n_epochs=300, valid_size=0,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None, convert=False):
    if seed is not None:
        torch.manual_seed(seed)

    # Create train/valid split
    if valid_size:
        indices = torch.randperm(len(train_set))
        train_indices = indices[:len(indices) - valid_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_indices = indices[len(indices) - valid_size:]
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Data loaders
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=4)
    if valid_size:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=4)
        valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=4)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=4)
        valid_loader = test_loader

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    timestamp = str(time.time())[:10]
    result_log = 'results{}.csv'.format(timestamp)
    # Start log
    with open(os.path.join(save, result_log), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    print("Starting training with {} epochs and batch size of {}".format(n_epochs, batch_size))

    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            convert=convert
        )

        with torch.no_grad():
            _, valid_loss, valid_error = test_epoch(
                model=model_wrapper,
                loader=valid_loader if valid_loader else test_loader,
                is_test=(not valid_loader),
                convert=convert
            )

        # Determine if model is the best
        if valid_loader and valid_error < best_error:
            best_error = valid_error
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, 'model{}.dat'.format(timestamp)))

        # Log results
        with open(os.path.join(save, result_log), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))

    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(save, 'model{}.dat'.format(timestamp))))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model,
        loader=test_loader,
        is_test=True,
        convert=convert
    )
    _, _, test_error = test_results
    with open(os.path.join(save, result_log), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)


def demo(data_path, save="./save/", n_epochs=300, batch_size=64, seed=None, convert=False, no_aug=False, valid_size=0):
    """
    A demo to show training of checkered convolutional neural networks.
    Replace the model and dataset with your own in this method.

    Args:
        data_path (str) - path to directory with your CIFAR dataset. Automatically downloads if not found.

        save (str) - path to save best model and training logs (default ./save/)
        n_epochs (int) - number of epochs to train for (default 300)
        batch_size (int) - size of minibatch (default 64)
        seed (int) - manually set the random seed (default False)
        convert (bool) - converts model to a checkered CNN (default False)
        no_aug (bool) - turns off data augmentations (default False)
        valid_size - how much of training set to use as validation data (default 0, uses test set)
    """

    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    if no_aug:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
    ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    train_set = datasets.CIFAR10(data_path, train=True, transform=train_transforms, download=True)
    test_set = datasets.CIFAR10(data_path, train=False, transform=test_transforms, download=False)

    # Model
    model = ResNet18(10)

    if convert:
        print("Converting to checkered CNN")
        model.apply(convert_to_checkered)

    print("Parameter count: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model=model, train_set=train_set, test_set=test_set, save=save,
          valid_size=valid_size, n_epochs=n_epochs, batch_size=batch_size, seed=seed, convert=convert)
    print('Done!')


"""
Args:
    --data_path (string) - path to the directory with your dataset (CIFAR10/CIFAR100)

To train a model:
python demo.py --data <path_to_data_dir>

To train a model as a CCNN:
python demo.py --data <path_to_data_dir> --convert

Replace the model and dataset with your own in the demo method.

Other args:
    --save (string) - directory to save training logs and model parameters (default ./save/)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 64)
    --seed (int) - manually set the random seed (default None)
    --convert (bool) - whether or not to convert model into CCNN (default False)
    --no_aug (bool) - turn off data augmentations (default False)
    --valid_size (int) - size of validation set (default 0, uses test set)
"""
if __name__ == '__main__':
    fire.Fire(demo)

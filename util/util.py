"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from email.policy import strict
import re
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image
import os
import argparse
import dill as pickle
import util.coco
import random
import torch.nn.functional as F

def DiffAugment(real_img, fake_img, label, policy=''):
    if policy:
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                real_img, fake_img, label = f(real_img, fake_img, label)
        real_img, fake_img = real_img.contiguous(), fake_img.contiguous()
    return real_img, fake_img, label


def rand_brightness(real_img, fake_img, label, strength=1.):
    real_img = real_img + (torch.rand(real_img.size(0), 1, 1, 1, dtype=real_img.dtype, device=real_img.device) - 0.5)*strength
    fake_img = fake_img + (torch.rand(fake_img.size(0), 1, 1, 1, dtype=fake_img.dtype, device=fake_img.device) - 0.5)*strength
    return real_img, fake_img, label


def rand_saturation(real_img, fake_img, label):
    real_img_mean = real_img.mean(dim=1, keepdim=True)
    real_img = (real_img - real_img_mean) * (torch.rand(real_img.size(0), 1, 1, 1, dtype=real_img.dtype, device=real_img.device) * 2) + real_img_mean
    fake_img_mean = fake_img.mean(dim=1, keepdim=True)
    fake_img = (fake_img - fake_img_mean) * (torch.rand(fake_img.size(0), 1, 1, 1, dtype=fake_img.dtype, device=fake_img.device) * 2) + fake_img_mean
    return real_img, fake_img, label


def rand_contrast(real_img, fake_img, label):
    real_img_mean = real_img.mean(dim=[1, 2, 3], keepdim=True)
    real_img = (real_img - real_img_mean) * (torch.rand(real_img.size(0), 1, 1, 1, dtype=real_img.dtype, device=real_img.device) + 0.5) + real_img_mean
    fake_img_mean = fake_img.mean(dim=[1, 2, 3], keepdim=True)
    fake_img = (fake_img - fake_img_mean) * (torch.rand(fake_img.size(0), 1, 1, 1, dtype=fake_img.dtype, device=fake_img.device) + 0.5) + fake_img_mean
    
    return real_img, fake_img, label

def rand_crop(img, fake,label, strength=0.5):
    b, _, h, w = img.shape
    size = (h, w)
    fake =  F.interpolate(fake, size=size, mode='bicubic')
    label =  F.interpolate(label, size=size, mode='nearest')
    size = (int(h*1.2), int(w*1.2))
    img_large = F.interpolate(img, size=size, mode='bicubic')
    fake_large = F.interpolate(fake, size=size, mode='bicubic')
    label_large = F.interpolate(label, size=size, mode='nearest')
    _, _, h_large, w_large = img_large.size()
    h_start, w_start = random.randint(0, (h_large - h)), random.randint(0, (w_large - w))
    # print(h_start, w_start)
    img_crop = img_large[:, :, h_start:h_start+h, w_start:w_start+w]
    fake_crop = fake_large[:, :, h_start:h_start+h, w_start:w_start+w]
    label_crop = label_large[:, :, h_start:h_start+h, w_start:w_start+w]
    assert img_crop.size() == img.size()
    is_crop = torch.rand([b, 1, 1, 1], device=img.device) < strength
    img = torch.where(is_crop, img_crop, img)
    fake = torch.where(is_crop, fake_crop, fake)
    label = torch.where(is_crop, label_crop, label)
    return img, fake,label


def rand_translation(real_img, fake_img, label, ratio=0.125):
    x = real_img
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    real_img_pad = F.pad(real_img, [1, 1, 1, 1, 0, 0, 0, 0])
    real_img = real_img_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    fake_img_pad = F.pad(fake_img, [1, 1, 1, 1, 0, 0, 0, 0])
    fake_img = fake_img_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    label_pad = F.pad(label, [1, 1, 1, 1, 0, 0, 0, 0])
    label = label_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return real_img, fake_img, label


# def rand_cutout(x, ratio=0.5):
#     cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
#     offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
#     offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
#     grid_batch, grid_x, grid_y = torch.meshgrid(
#         torch.arange(x.size(0), dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
#     )
#     grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
#     grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
#     mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
#     mask[grid_batch, grid_x, grid_y] = 0
#     x = x * mask.unsqueeze(1)
#     return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'crop': [rand_crop],
}

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here


def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False, is_img = False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    if is_img:
        image_pil.save(image_path)
    else:
        image_pil.save(image_path.replace('.jpg', '.png'))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights, strict=False)
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

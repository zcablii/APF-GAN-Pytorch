# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
import random

def DiffAugment(real_img, fake_img, label, policy=''):
    if policy:
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                real_img, fake_img, label = f(real_img, fake_img, label)
        real_img, fake_img = real_img.contiguous(), fake_img.contiguous()
    return real_img, fake_img, label


def rand_brightness(real_img, fake_img, label):
    real_img = real_img + (torch.rand(real_img.size(0), 1, 1, 1, dtype=real_img.dtype, device=real_img.device) - 0.5)
    fake_img = fake_img + (torch.rand(fake_img.size(0), 1, 1, 1, dtype=fake_img.dtype, device=fake_img.device) - 0.5)
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

def rand_crop(img, fake,label):
    b, _, h, w = img.shape
    img_large = torch.nn.functional.interpolate(img, scale_factor=1.2, mode='bicubic')
    fake_large = torch.nn.functional.interpolate(fake, scale_factor=1.2, mode='bicubic')
    label_large = torch.nn.functional.interpolate(label, scale_factor=1.2, mode='nearest')
    _, _, h_large, w_large = img_large.size()
    h_start, w_start = random.randint(0, (h_large - h)), random.randint(0, (w_large - w))
    # print(h_start, w_start)
    img_crop = img_large[:, :, h_start:h_start+h, w_start:w_start+w]
    fake_crop = fake_large[:, :, h_start:h_start+h, w_start:w_start+w]
    label_crop = label_large[:, :, h_start:h_start+h, w_start:w_start+w]
    assert img_crop.size() == img.size()
    is_crop = torch.rand([b, 1, 1, 1], device=img.device) < 0.5
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

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_h_sz, grid_w_sz):
    grid_h = np.arange(grid_h_sz, dtype=np.float16)
    grid_w = np.arange(grid_w_sz, dtype=np.float16)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_sz, grid_w_sz])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, use_pos=False, use_pos_proj=False, add_noise = False, opt=None):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        self.add_noise = add_noise
        self.use_pos = use_pos
        self.use_pos_proj = use_pos_proj
        if opt.use_seg_noise:
            k = opt.use_seg_noise_kernel
            self.seg_noise_var = nn.Conv2d(label_nc, norm_nc, k, padding=(k-1)//2)
            self.seg_noise_var.weight.data.fill_(0)
            self.seg_noise_var.bias.data.fill_(0)
            print('use seg noise var!!!! initialize all 0, use kernel:', opt.use_seg_noise_kernel)
        # if use_pos:
        #     print('use_pos true!!!!!!!!')
        #     if use_pos_proj:
        #         print('use_pos_proj true!!!!!!!!')

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        if self.add_noise:
            self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)
        self.pos_embed = None
        if use_pos_proj:
            self.pos_proj = nn.Conv2d(nhidden, nhidden, kernel_size=1)

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.opt = opt

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        if self.opt.use_seg_noise:
            seg = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
            noise = self.seg_noise_var(seg)
            added_noise = (torch.randn(noise.shape[0], 1, noise.shape[2], noise.shape[3]).cuda() * noise)
            normalized = self.param_free_norm(x + added_noise)
        elif self.add_noise:
            added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)
            normalized = self.param_free_norm(x + added_noise)
        else: 
            normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        if self.use_pos: # default with True
            if self.pos_embed is None:
                B, C, H, W = actv.size()
                pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(C, H, W)).to(actv.device)
                self.pos_embed = pos_embed.permute(0, 1).reshape(1, C, H, W)
            if self.use_pos_proj: # default with True
                actv += self.pos_proj(self.pos_embed.to(torch.float16))
            else:
                actv += self.pos_embed

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return 
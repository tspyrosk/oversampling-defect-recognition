# Contains code from:
# https://github.com/ajbrock/BigGAN-PyTorch/
# https://github.com/nogu-atsu/small-dataset-image-generation
# https://github.com/apple2373/MetaIRNet

import math
import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import os
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler


def power_iteration(W, u_, update=True, eps=1e-12):
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs

class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))
    
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
     return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
     
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv     
        return self.weight / svs[0]

class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, dilation=1, groups=1, bias=True, 
                         num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                                         padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride, 
                                        self.padding, self.dilation, self.groups)
        
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                             num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)

class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
                             cross_replica=False, mybn=False, norm_style='bn',):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn
        self.norm_style = norm_style
        
        if self.cross_replica:
            pass
        elif self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var',  torch.ones(output_size)) 
        
        
    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.mybn or self.cross_replica:
            return self.bn(x, gain=gain, bias=bias)
        else:
            if self.norm_style == 'bn':
                out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                                                    self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                                                    self.training, 0.1, self.eps)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias
    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s +=' cross_replica={cross_replica}'
        return s.format(**self.__dict__)

class bn(nn.Module):
    def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                                cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.output_size= output_size
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn
        
        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)    
        elif mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        else:     
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var',  torch.ones(output_size))
        
    def forward(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1,-1,1,1)
            bias = self.bias.view(1,-1,1,1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                                                    self.bias, self.training, self.momentum, self.eps)

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                             which_conv=nn.Conv2d, which_bn=bn, activation=None, 
                             upsample=None):
        super(GBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                                                         kernel_size=1, padding=0)
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x

class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        self.gamma = P(torch.tensor(0.), requires_grad=True)
    def forward(self, x, y=None):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

def upsample_deterministic(x,upscale):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, upscale, -1, upscale).reshape(x.size(0), x.size(1), x.size(2)*upscale, x.size(3)*upscale)

def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                             'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
                             'upsample' : [True] * 7,
                             'resolution' : [8, 16, 32, 64, 128, 256, 512],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,10)}}
    arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
                             'upsample' : [True] * 6,
                             'resolution' : [8, 16, 32, 64, 128, 256],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,9)}}
    arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
                             'upsample' : [True] * 5,
                             'resolution' : [8, 16, 32, 64, 128],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,8)}}
    arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
                             'out_channels' : [ch * item for item in [16, 8, 4, 2]],
                             'upsample' : [True] * 4,
                             'resolution' : [8, 16, 32, 64],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,7)}}
    arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
                             'out_channels' : [ch * item for item in [4, 4, 4]],
                             'upsample' : [True] * 3,
                             'resolution' : [8, 16, 32],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,6)}}

    return arch

class Generator(nn.Module):
    def __init__(self, G_ch=96, dim_z=120, bottom_width=4, resolution=128,
                             G_kernel_size=3, G_attn='64', n_classes=1000,
                             num_G_SVs=1, num_G_SV_itrs=1,
                             G_shared=True, shared_dim=128, hier=True,
                             cross_replica=False, mybn=False,
                             G_activation=nn.ReLU(inplace=True),
                             G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                             BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                             G_init='ortho', skip_init=True, no_optim=True,
                             G_param='SN', norm_style='bn',
                             **kwargs):
        super(Generator, self).__init__()
        self.ch = G_ch
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.G_shared = G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        self.hier = hier
        self.cross_replica = cross_replica
        self.mybn = mybn
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.norm_style = norm_style
        self.BN_eps = BN_eps
        self.SN_eps = SN_eps
        self.fp16 = G_fp16
        self.arch = G_arch(self.ch, self.attention)[resolution]

        if self.hier:
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            self.dim_z = self.z_chunk_size *  self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            
        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                                 else self.which_embedding)
        self.which_bn = functools.partial(ccbn,
                                                    which_linear=bn_linear,
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn,
                                                    input_size=(self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes),
                                                    norm_style=self.norm_style,
                                                    eps=self.BN_eps)



        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared else layers.identity())
        self.linear = self.which_linear(self.dim_z // self.num_slots, self.arch['in_channels'][0] * (self.bottom_width **2))

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                                         out_channels=self.arch['out_channels'][index],
                                                         which_conv=self.which_conv,
                                                         which_bn=self.which_bn,
                                                         activation=self.activation,
                                                         upsample=(functools.partial(upsample_deterministic, upscale=2)
                                                                             if self.arch['upsample'][index] else None))]]

            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.output_layer = nn.Sequential(bn(self.arch['out_channels'][-1],
                                             cross_replica=self.cross_replica,
                                             mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], 3))

        if not skip_init:
            self.init_weights()

        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print('Using fp16 adam in G...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                                     betas=(self.B1, self.B2), weight_decay=0,
                                                     eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                     betas=(self.B1, self.B2), weight_decay=0,
                                                     eps=self.adam_eps)

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) 
                    or isinstance(module, nn.Linear) 
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, z, y):
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)
            
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, ys[index])
                
        return torch.tanh(self.output_layer(h))

def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x

from torchvision import models


def create_vgg16(dict_path=None):
    model = models.vgg16(pretrained=False)
    if (dict_path != None):
        model.load_state_dict(torch.load(dict_path))
    return model

class Vgg16PerceptualLoss(torch.nn.Module):
    def __init__(self, perceptual_indices = [1,3,6,8,11,13,15,18,20,22] ,loss_func="l1",requires_grad = False):
        super(Vgg16PerceptualLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True)
        torch.save(vgg_pretrained_features.state_dict(), "vgg_weights.pth")

        vgg_pretrained_features = create_vgg16("vgg_weights.pth").features.eval()
        max_layer_idx = max(perceptual_indices)
        self.perceptual_indices = set(perceptual_indices)
        self.vgg_partial = torch.nn.Sequential(*list(vgg_pretrained_features.children())[0:max_layer_idx])
        
        if loss_func == "l1":
            self.loss_func = F.l1_loss
        elif loss_func == "l2":
            self.loss_func = F.mse_loss
        elif loss_func == "sl1":
            self.loss_func = F.smooth_l1_loss
        else:
            raise NotImpementedError(loss_func)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self,batch):
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std
    
    def rescale(self,batch,lower,upper):
        return  (batch - lower)/(upper - lower)

    def forward_img(self, h):      
        intermidiates = []
        for i,layer in enumerate(self.vgg_partial):
            h = layer(h)
            if i in self.perceptual_indices:
                intermidiates.append(h)
        return intermidiates    

    def forward(self, img1, img2, img1_minmax=(0,1),img2_minmax=(0,1), apply_imagenet_norm = True):
        if img1_minmax!=(0,1):
            img1 = self.rescale(img1,img1_minmax[0],img1_minmax[1])
        if img2_minmax!=(0,1):
            img2 = self.rescale(img2,img2_minmax[0],img2_minmax[1])
            
        if apply_imagenet_norm:
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
        
        losses = []
        for img1_h,img2_h in zip(self.forward_img(img1),self.forward_img(img2)):
            losses.append(self.loss_func(img1_h,img2_h))
        
        return losses

class AdaBIGGANLoss(nn.Module):
    def __init__(self,perceptual_loss = "vgg",
                 scale_per=0.001,
                 scale_emd=0.1,
                 scale_reg=0.02,
                 normalize_img = True,
                 normalize_per = False,
                 dist_per = "l1",
                 dist_img = "l1",
                ):
        super(AdaBIGGANLoss,self).__init__()
        if perceptual_loss == "vgg":
            self.perceptual_loss =  Vgg16PerceptualLoss(loss_func=dist_per)
        else:
            self.perceptual_loss =  perceptual_loss
        self.scale_per = scale_per
        self.scale_emd = scale_emd
        self.scale_reg = scale_reg
        self.normalize_img = normalize_img
        self.normalize_perceptural = normalize_per
        self.dist_img = dist_img
        
    def earth_mover_dist(self,z):
        dim_z = z.shape[1]
        n = z.shape[0]
        t = torch.randn((n * 10,dim_z),device=z.device)
        dot = torch.matmul(z, t.permute(-1, -2))
        
        dist = torch.sum(z ** 2, dim=1, keepdim=True) - 2 * dot + torch.sum(t ** 2, dim=1)
        
        return torch.mean(dist.min(dim=0)[0]) + torch.mean(dist.min(dim=1)[0])

    def l1_reg(self,W):
        return torch.mean( W ** 2 )

    def forward(self,x,y,z,W):
        if self.dist_img == "l1":
            image_loss = F.l1_loss(x, 2.0*(y - 0.5) )
        elif self.dist_img == "l2":
            image_loss = F.mse_loss(x, 2.0*(y - 0.5) )
        elif self.dist_img == "sl1":
            image_loss = F.smooth_l1_loss(x, 2.0*(y - 0.5) )
        else:
            raise NotImpementedError(loss_func)
            
        if self.normalize_img:
            loss = image_loss/image_loss.item()
        else:
            loss = image_loss
        
        for ploss in self.perceptual_loss(img1=x,img2=y,img1_minmax=(-1,1),img2_minmax=(0,1)):
            if self.normalize_perceptural:
                loss += self.scale_per*ploss/ploss.item()
            else:
                loss += self.scale_per*ploss
            
        loss += self.scale_emd*self.earth_mover_dist(z)
        
        loss += self.scale_reg*self.l1_reg(W)
        
        return  loss

class AdaBIGGAN(nn.Module):
    def __init__(self,generator, dataset_size, embed_dim=120, shared_embed_dim = 128,cond_embed_dim = 20,embedding_init="zero",embedding_class_init="mean"):
        super(AdaBIGGAN,self).__init__()
        self.generator = generator
            
        self.embeddings = nn.Embedding(dataset_size, embed_dim)
        print("model embedding_init",embedding_init)
        if embedding_init == "zero":
            self.embeddings.from_pretrained(torch.zeros(dataset_size,embed_dim),freeze=False)
        elif embedding_init == "random":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        in_channels = self.generator.blocks[0][0].conv1.in_channels
        self.bsa_linear_scale = torch.nn.Parameter(torch.ones(in_channels,))
        self.bsa_linear_bias = torch.nn.Parameter(torch.zeros(in_channels,))
        
        self.linear = nn.Linear(1, shared_embed_dim, bias=False)
        print("model embedding_class_init",embedding_class_init)
        if embedding_class_init =="mean":
            init_weight = generator.shared.weight.mean(dim=0,keepdim=True).transpose(1,0)
            assert self.linear.weight.data.shape == init_weight.shape
            self.linear.weight.data  = init_weight
            del generator.shared
        elif embedding_class_init == "zero":
            self.linear.weight.data  = torch.zeros(self.linear.weight.data.shape)
        elif  embedding_class_init =="random":
            scale = 0.001 ** 0.5
            fan_out = self.linear.weight.size()[0]
            fan_in = self.linear.weight.size()[1]
            import numpy as np
            std = scale * np.sqrt(2. / fan_in)
            dtype = self.linear.weight.data.dtype
            self.linear.weight.data = torch.tensor(np.random.normal(loc=0.0,scale=std,size=(fan_out,fan_in)),dtype=dtype)
        else:
            raise NotImplementedError()
            
        self.set_training_parameters()
                
    def forward(self, z):
        y = torch.ones((z.shape[0], 1),dtype=torch.float32,device=z.device)
        y = self.linear(y)

        if self.generator.hier:
            zs = torch.split(z, self.generator.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            raise NotImplementedError("I don't implement this case")
            ys = [y] * len(self.generator.blocks)

        h = self.generator.linear(z)
        h = h.view(h.size(0), -1, self.generator.bottom_width, self.generator.bottom_width)
        
        h = h*self.bsa_linear_scale.view(1,-1,1,1) + self.bsa_linear_bias.view(1,-1,1,1) 
        
        for index, blocklist in enumerate(self.generator.blocks):
            for block in blocklist:
                h = block(h, ys[index])

        return torch.tanh(self.generator.output_layer(h))
    

    
    def set_training_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
            
        named_params_requires_grad = {}
        named_params_requires_grad.update(self.batch_stat_gen_params())
        named_params_requires_grad.update(self.linear_gen_params())
        named_params_requires_grad.update(self.bsa_linear_params())
        named_params_requires_grad.update(self.calss_conditional_embeddings_params())
        named_params_requires_grad.update(self.emebeddings_params())
        
        for name,param in named_params_requires_grad.items():
            param.requires_grad = True
            
    def batch_stat_gen_params(self):
        named_params = {}
        for name,value in self.named_modules():
            if name.split(".")[-1] in ["gain","bias"]:
                for name2,value2 in  value.named_parameters():
                    name = name+"."+name2
                    params = value2
                    named_params[name] = params
                    
        return named_params
       
    def linear_gen_params(self):
        return {"generator.linear.weight":self.generator.linear.weight,
                       "generator.linear.bias":self.generator.linear.bias}

    def bsa_linear_params(self):
        return {"bsa_linear_scale":self.bsa_linear_scale,"bsa_linear_bias":self.bsa_linear_bias}

    def calss_conditional_embeddings_params(self):
        return {"linear.weight":self.linear.weight}


    def emebeddings_params(self):
        return  {"embeddings.weight":self.embeddings.weight}

def setup_optimizer(model,lr_g_batch_stat,lr_g_linear,lr_bsa_linear,lr_embed,lr_class_cond_embed,step,step_facter=0.1):
    params = []
    params.append({"params":list(model.batch_stat_gen_params().values()), "lr":lr_g_batch_stat})
    if lr_g_linear > 0:
        params.append({"params":list(model.linear_gen_params().values()), "lr":lr_g_linear })
    else:
        for p in model.linear_gen_params().values():
            p.requires_grad = False   
    params.append({"params":list(model.bsa_linear_params().values()), "lr":lr_bsa_linear })
    params.append({"params":list(model.emebeddings_params().values()), "lr": lr_embed })
    params.append({"params":list(model.calss_conditional_embeddings_params().values()), "lr":lr_class_cond_embed})
    
    optimizer = optim.Adam(params, lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=step_facter)
    return optimizer,scheduler
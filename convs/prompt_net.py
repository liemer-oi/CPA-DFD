import copy
import this
import numpy as np
import torch
from torch import nn
from torch.distributions import relaxed_bernoulli
import torch.nn.functional as F
from convs.linears import SimpleLinear, CosineLinear, SplitCosineLinear

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from convs.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from convs.convit import ClassAttention 
from convs.convit import Block as ConBlock

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_KPrompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p 

class Baseline_Net(nn.Module):
    def __init__(self,args):
        super(Baseline_Net,self).__init__()
        self.args=args
        self.dataset_name=args["dataset"]
        self.fc = None
        self._device = args['device'][0]
        
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=False, **model_kwargs)

        # self.task_tokens = copy.deepcopy(self.image_encoder.cls_token)
        
        # for name, param in self.image_encoder.named_parameters():
        #     param.requires_grad=False
        #     param.grad=None
    
    def update_fc(self, nb_classes, nextperiod_initialization=None):
        feature_dim = self.image_encoder.embed_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


    def forward_wo_p(self, image):
        image_features = image_features = self.image_encoder(image, returnbeforepool=True)
        feature=image_features[:,0,:]
        out = self.fc(feature)
        out.update({"features": feature})
        return out

    
    def forward(self, image, gen_p):
        image_features = self.image_encoder(image, instance_tokens=gen_p[0],second_pro=gen_p[1], returnbeforepool=True)
        feature=image_features[:,0,:]

        out = self.fc(feature)
        out.update({"features": feature})
        return out
    
    def get_feature(self,image, gen_p):
        # i是当前任务的prompt
        image_features = self.image_encoder(image, instance_tokens=gen_p[0],second_pro=gen_p[1], returnbeforepool=True)
        feature=image_features[:,0,:]
        return feature
    
    def fix_branch_layer(self):
        for param in self.vitprompt_1.parameters():
            param.requires_grad=False
            param.grad=None
            
        for param in self.vitprompt_2.parameters():
            param.requires_grad=False
            param.grad=None

        
class ViT_KPrompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)

    def forward(self, x, instance_tokens=None, second_pro=None, returnbeforepool=False,gen_pro=None, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if gen_pro is None:
            if instance_tokens is not None and instance_tokens.shape[0]!=16:
                instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

            x = x + self.pos_embed.to(x.dtype)
            # 在第1个block前插prompt
            if instance_tokens is not None:
                x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1) # prompt插在token和patch embedding之间
            x = self.pos_drop(x)

            x=self.blocks[:5](x)
            # 在第6个block前插prompt
            if second_pro is not None:
                second_pro=second_pro.to(x.dtype)+torch.zeros(x.shape[0],1,x.shape[-1],dtype=x.dtype,device=x.device)
                x = torch.cat([x[:,:1+instance_tokens.shape[1],:], second_pro, x[:,1+instance_tokens.shape[1]:,:]], dim=1)
            x=self.blocks[5:](x)
        else:
            for i in range(len(instance_tokens)):
                if instance_tokens[i].shape[1]==768:
                    instance_tokens[i]=instance_tokens[i].to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                else:
                    instance_tokens[i]=instance_tokens[i].to(x.dtype)
            
            x = x + self.pos_embed.to(x.dtype)
            x = torch.cat([x[:,:1,:], instance_tokens[0], x[:,1:,:]], dim=1)
            x = self.pos_drop(x)
            x=self.blocks[0](x)
            for i in range(len(instance_tokens)-1):
                x = torch.cat([x[:,:1+instance_tokens[0].shape[1]*(i+1),:], instance_tokens[i+1], x[:,1+instance_tokens[0].shape[1]*(i+1):,:]], dim=1)
                x=self.blocks[i+1](x)
            x=self.blocks[len(instance_tokens):](x)

        if returnbeforepool == True:
            return x
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x
    
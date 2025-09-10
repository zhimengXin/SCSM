import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import torchvision as tv

import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from scsm.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from scsm.modeling.roi_heads import build_roi_heads
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from torchvision import transforms

#import  from MMDetection start
from mmcv.cnn import ConvModule
#from mmcv.runner import BaseModule
from detectron2.structures import Boxes

# from mmcv.cnn import (build_activation_layer, build_conv_layer,
#                       build_norm_layer, xavier_init)

from mmengine.model import caffe2_xavier_init, constant_init

#from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_positional_encoding)
#from mmdet.models.utils.builder import TRANSFORMER

from torch.nn.init import normal_
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

from torchvision import transforms
# from torchvision.transforms import v2

import os

import os
#import from MMDetection end

#for plot heat map
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform

from scsm.modeling.meta_arch.models_mamba import create_block


__all__ = ["GeneralizedRCNN"]


def cutmix(images, targets, alpha=1.0):
    batch_size = len(images)
    lam = torch.tensor([np.random.beta(alpha, alpha) for _ in range(batch_size)]).to(images[0].device)
    indices = torch.randperm(batch_size).to(images[0].device)

    mixed_images = []
    mixed_targets = []

    for i in range(batch_size):
        image_a, image_b = images[i], images[indices[i]]
        target_a, target_b = targets[i], targets[indices[i]]

        height = min(image_a.shape[1], image_b.shape[1])
        width = min(image_a.shape[2], image_b.shape[2])

        cut_ratio = np.sqrt(1.0 - lam[i].cpu().detach().numpy())
        
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        image_a[:, bby1:bby2, bbx1:bbx2] = image_b[:, bby1:bby2, bbx1:bbx2]

        mixed_images.append(image_a)
        mixed_targets.append((target_a, target_b, lam[i]))

    return mixed_images, mixed_targets

def mixup_feature(images, alpha=1.0):
    batch_size = images["res4"].shape[0]
    lam = torch.tensor([np.random.beta(alpha, alpha) for _ in range(batch_size)]).to(images["res4"].device)
    indices = torch.randperm(batch_size).to(images["res4"].device)

    mixed_images = {}

    for key, tensor in images.items():
        height = min(tensor.shape[2], tensor[indices].shape[2])
        width = min(tensor.shape[3], tensor[indices].shape[3])

        mixed_tensor = lam.view(-1, 1, 1, 1) * tensor 
        mixed_tensor[:,:,:height, :width] += (1.0 - lam.view(-1, 1, 1, 1)) * tensor[indices][:,:,:height, :width]

        mixed_tensor = lam.view(-1, 1, 1, 1) * tensor + (1.0 - lam.view(-1, 1, 1, 1)) * tensor[indices]
        mixed_images[key] = mixed_tensor

    return mixed_images

def mixup(images, alpha=1.0):
    batch_size = len(images)
    lam = torch.tensor([np.random.beta(alpha, alpha) for _ in range(batch_size)]).to(images[0].device)
    indices = torch.randperm(batch_size).to(images[0].device)

    mixed_images = []
    mixed_targets = []

    for i in range(batch_size):
        image_a, image_b = images[i], images[indices[i]]
        lam = lam.to(image_a.device)
        if image_a.shape[1] == image_b.shape[1] and image_a.shape[2] == image_b.shape[2]:
            mixed_image = lam[i] * image_a + (1 - lam[i]) * image_b
        else:
            height = min(image_a.shape[1], image_b.shape[1])
            width = min(image_a.shape[2], image_b.shape[2])

            mixed_image = lam[i]* image_a
            mixed_image[:, :height, :width] += (1 - lam[i]) * image_b[:, :height, :width]


        mixed_images.append(mixed_image)

    return mixed_images

def cutout(images, length):
    masked_images = []
    batch_size = len(images)
    for i, image in enumerate(images):
        channels, height, width = image.shape

        mask = np.ones((height, width), np.float32)
        y = np.random.randint(height)
        x = np.random.randint(width)
        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)
        mask[ y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).expand_as(image).to(image.device)
        masked_image = image * mask

        masked_images.append(masked_image)

    return masked_images

def cutout_feat(images, length):
    masked_images = {}
    for key, image in images.items():
        _,channels, height, width = image.shape

        mask = np.ones((height, width), np.float32)
        y = np.random.randint(height)
        x = np.random.randint(width)
        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)
        mask[ y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).expand_as(image).to(image.device)
        masked_image = image * mask

        masked_images[key] = masked_image

    return masked_images

######adapter:

class AdapterForward(nn.Module):
    def __init__(self, dim, mid_dim, dim_out=None, act="relu"):
        super().__init__()
        dim_out = dim
        assert act in ["relu", "sig", "silu", "ident", "default"]
        af = None
        if act == "relu" or act == "default":
            af = nn.ReLU()
        elif act == "sig":
            af = nn.Sigmoid()
        elif act == "silu":
            af = nn.SiLU()
        elif act == "ident":
            af = nn.Identity()
        else:
            print("act")
            raise
        self.adp = nn.Sequential(nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1), af, nn.Conv2d(mid_dim, dim_out,  kernel_size=3, padding=1))

        nn.init.kaiming_normal_(self.adp[0].weight)
        nn.init.kaiming_normal_(self.adp[2].weight)

    def forward(self, x):
        return (self.adp(x) + x)



@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        #self.bnclean = torch.nn.BatchNorm2d
        self.bn_attack = torch.nn.BatchNorm2d
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        
        self.mamba = create_block(d_model=1024)

        #self.onlymamba = cfg.MODEL.BACKBONE.WITHMAMBA

        self.with_adapter = cfg.MODEL.BACKBONE.CNN
        if self.with_adapter:
            self.adapter = AdapterForward(1024, 360)
        # else:
        #     self.adapter = None

        if cfg.MODEL.BACKBONE.DATA_AUG_TYPE == "RandomErasing":
            self.transform = transforms.Compose([
                transforms.RandomErasing()
            ])
        else:
            self.transform = None
    
        if cfg.MODEL.BACKBONE.FEATURE_AUG_TYPE == "RandomErasing":
            self.transform_feat = transforms.Compose([
                transforms.RandomErasing()
            ])
        else:
            self.transform_feat = None

        #self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        #self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_[self.roi_heads.in_features[-1]].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_[self.roi_heads.in_features[-1]].channels, bias=True)
        self.features = []

        self.WithExtConv = False
        self.ExtConvLayer = 1
        if self.WithExtConv:
            hidden_dim = 1024
            input_dim = self._SHAPE_['res4'].channels
            out_channels = hidden_dim
            for i in range(self.ExtConvLayer):
                if i == 0:
                    in_channels = input_dim
                else:
                    in_channels = hidden_dim

                if i == self.ExtConvLayer - 1:
                    out_channels = input_dim
                else:
                    out_channels = hidden_dim

                setattr(self, f'ext_conv_{i}', nn.Conv2d(in_channels, out_channels, 3, 1, 1))
                
                weight_init.c2_xavier_fill(getattr(self, f'ext_conv_{i}'))

        
        if cfg.MODEL.BACKBONE.WITHECEA:
            # add transofomer encoder
            
            in_channels_trans = 2048
            num_query=300
            num_feature_levels=4

            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True)
            #in_channels = 1024
            in_channels = self._SHAPE_[self.roi_heads.in_features[-1]].channels
            out_channels = 256
            
            
            norm = ""
            use_bias = norm == ""
            self.lateral_conv_in = Conv2d( in_channels, out_channels, kernel_size=1, bias=use_bias )
            self.lateral_conv_out = Conv2d( out_channels, in_channels, kernel_size=1, bias=use_bias ) 
            
            weight_init.c2_xavier_fill(self.lateral_conv_in)
            weight_init.c2_xavier_fill(self.lateral_conv_out)
            
            encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=4,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256, num_levels = len(self._SHAPE_), num_points = 4 ),#num_levels必须等于num_outs
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')))
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform')
            
            num_outs = len(self._SHAPE_) 
            self.num_feature_levels = out_channels
            
            self.encoder = build_transformer_layer_sequence(encoder)
            self.embed_dims = self.encoder.embed_dims
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.positional_encoding = build_positional_encoding( positional_encoding )
            
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))

############################################################################################################################
        """
            - 2024.6.28
        """
        self.SCSM = cfg.MODEL.BACKBONE.WITHSCSM
        if self.SCSM:
            # add transofomer encoder

            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True)
            #in_channels = 1024
            in_channels = self._SHAPE_[self.roi_heads.in_features[-1]].channels
            out_channels = 256
            
            norm = ""
            use_bias = norm == ""
            self.lateral_conv_in = Conv2d( in_channels, out_channels, kernel_size=1, bias=use_bias )
            self.lateral_conv_out = Conv2d( out_channels, in_channels, kernel_size=1, bias=use_bias ) 
            
            weight_init.c2_xavier_fill(self.lateral_conv_in)
            weight_init.c2_xavier_fill(self.lateral_conv_out)
            
            encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256, num_levels = len(self._SHAPE_), num_points = 4 ),#num_levels必须等于num_outs
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')))
            self.num_feature_levels = out_channels
            self.encoder = build_transformer_layer_sequence(encoder)
            # self.mamba_block = create_block(d_model=256)
            self.mamba_pooled = 256 # 16*16
            for i in range(1):
                setattr(self, f'encoder_{i}', build_transformer_layer_sequence(encoder))
                # setattr(self, f'mamba_block_{i}', create_block(d_model=256))
                setattr(self, f'mamba_block_{i}', create_block(d_model=self.mamba_pooled))

            self.embed_dims = self.encoder.embed_dims
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.positional_encoding = build_positional_encoding( positional_encoding )
            
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))
            
############################################################################################################################

############################################################################################################################
        """
            - 2024.7.9
            - SFA+SA
        """
        from scsm.modeling.meta_arch.myTransformer import TransformerEncoderLayer
        self.SFASA = cfg.MODEL.BACKBONE.WITHSFASA
        if self.SFASA:
            # add transofomer encoder

            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True)
            #in_channels = 1024
            in_channels = self._SHAPE_[self.roi_heads.in_features[-1]].channels
            out_channels = 256
            
            norm = ""
            use_bias = norm == ""
            self.lateral_conv_in = Conv2d( in_channels, out_channels, kernel_size=1, bias=use_bias )
            self.lateral_conv_out = Conv2d( out_channels, in_channels, kernel_size=1, bias=use_bias ) 
            
            weight_init.c2_xavier_fill(self.lateral_conv_in)
            weight_init.c2_xavier_fill(self.lateral_conv_out)
            
            encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256, num_levels = len(self._SHAPE_), num_points = 4 ),#num_levels必须等于num_outs
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')))
            self.num_feature_levels = out_channels
            self.encoder = build_transformer_layer_sequence(encoder)

            for i in range(3):
                setattr(self, f'encoder_{i}', build_transformer_layer_sequence(encoder))
                setattr(self, f'transformer_block_{i}', TransformerEncoderLayer(256, 4, 1024))

            self.embed_dims = self.encoder.embed_dims
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.positional_encoding = build_positional_encoding( positional_encoding )
            
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))
            
############################################################################################################################
############################################################################################################################
        """
            - 2024.7.15
            - SFA+SENet
        """
        from scsm.modeling.meta_arch.myTransformer import SE_Block
        self.SFASENet = cfg.MODEL.BACKBONE.WITHSFASENet
        if self.SFASENet:
            # add transofomer encoder

            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True)
            #in_channels = 1024
            in_channels = self._SHAPE_[self.roi_heads.in_features[-1]].channels
            out_channels = 256
            
            norm = ""
            use_bias = norm == ""
            self.lateral_conv_in = Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias )
            self.lateral_conv_out = Conv2d(out_channels, in_channels, kernel_size=1, bias=use_bias ) 
            
            weight_init.c2_xavier_fill(self.lateral_conv_in)
            weight_init.c2_xavier_fill(self.lateral_conv_out)
            
            encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256, num_levels = len(self._SHAPE_), num_points = 4 ),#num_levels必须等于num_outs
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')))
            self.num_feature_levels = out_channels
            self.encoder = build_transformer_layer_sequence(encoder)

            for i in range(3):
                setattr(self, f'encoder_{i}', build_transformer_layer_sequence(encoder))
                setattr(self, f'se_block_{i}', SE_Block(256, 4))

            self.embed_dims = self.encoder.embed_dims
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.positional_encoding = build_positional_encoding( positional_encoding )
            
            self.reference_points = nn.Linear(self.embed_dims, 2)
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))
            
############################################################################################################################


        self.to(self.device) 
            
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")
        if cfg.MODEL.BACKBONE.FREEZE_ECEA:
            for p in self.encoder.parameters():
                p.requires_grad = False
            
            # for p in self.reference_points.requires_grad_:
            #     p.requires_grad = False
                
            # for p in self.lateral_conv_in:
            #     p.requires_grad = False
                
            # for p in self.lateral_conv_out:
            #     p.requires_grad = False
                
            # for p in self.level_embeds:
            #     p.requires_grad = False
                
            # for p in self.positional_encoding:
            #     p.requires_grad = False
                
            print("froze ECEA parameters")

        

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)    
          
    # def forward(self, batched_inputs):
    #     if not self.training:
    #         return self.inference(batched_inputs)
    #     assert "instances" in batched_inputs[0]
    #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
    #     proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
    #     losses = {}
    #     losses.update(detector_losses)
    #     losses.update(proposal_losses)
    #     return losses
    

    def forward(self, batched_inputs, is_attack = False, iter= None):
        if not self.training:
            return self.inference(batched_inputs, iter = iter)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances, is_attack=is_attack, iter = iter)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses#, gt_roi_features

    def generate_adversarial_examples(self, batched_inputs):
               
        batched_inputs_new = batched_inputs.copy()
        for input_data_new, input_data in zip( batched_inputs_new , batched_inputs):
            input_data_new['image'] = input_data['image'].to(torch.float).clone().detach()
            input_data_new['image'].to('cuda')
            input_data_new['image'].requires_grad = True            

        return batched_inputs_new
        

    def attack_samples(self, batched_inputs, inputs_grad, epsilon=0.1, alpha=0.01):
        # 使用基于梯度的攻击方法扰动输入数据，例如 FGSM (Fast Gradient Sign Method)
        perturbed_inputs = batched_inputs.copy()
        for perturbed_input, input, input_grad in zip(perturbed_inputs, batched_inputs, inputs_grad):
           
            original_image = input['image'].permute(1, 2, 0).detach().numpy()
           
            perturbed_input['image'] = input['image'] + epsilon * torch.sign(input_grad)  # 扰动输入数据
            # perturbed_inputs = inputs + epsilon * torch.sign(inputs_grad)  # 扰动输入数据
            # perturbed_input['image'] = torch.clamp( perturbed_input['image'], 0.0, 1.0)  # 确保像素值在合适的范围内
            adversarial_image = np.uint8(perturbed_input['image'].squeeze().permute(1, 2, 0).detach().numpy())
            # adversarial_image = torch.Tensor(adversarial_image)
            # adversarial_image = torch.clamp(adversarial_image, 0, 1)
            if self.cfg.SAVE_VIS_PLT:
                plt.subplot(1, 2, 1)
                plt.imshow(np.uint8(original_image))
                plt.title("Original Image")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(np.uint8(adversarial_image))
                plt.title("Adversarial Image")
                plt.axis("off")
                file_name = input['file_name'].split('/')[-1].split('.')[0]
                img_save_path = os.path.join(self.cfg.OUTPUT_DIR, file_name + '_Image_' + '.png')
                plt.savefig(img_save_path,dpi=500)
                plt.close()
                
        return perturbed_inputs

    def c_p_norm(x, p_norm_bound):
        p_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
        if p_norm > p_norm_bound:
            mask = True
        # scaling_factor = p_norm_bound / p_norm
        # scaling_factor = scaling_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # x[mask] = x[mask] * scaling_factor[mask]
        return mask

    def inference(self, batched_inputs, iter = None):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None, iter=iter)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def data_transforms(self, image):
        # Resize the image to 256x256
        #image = image.resize_((3, 224, 224))
        image = tv.transforms.Resize((224,224))(image)
        # Normalize the image
        image = image.float() / 255
        # Add a dimension to the image (batch_size, color, height, width)
        image = image.permute(1, 2, 0).cpu()
        image = image[:,:, [2,1,0]] # BGR==>RGB
        
        return image
    
    def visualize_attention(self, imageOrg, att_maps, S, n_top_attr, attr_name, attr,save_path=None, is_top=True):          #alphas_1: [bir]     alphas_2: [bi]

        image_size = 224         #one side of the img
        att_maps = att_maps.squeeze(0)
        att_maps = tv.transforms.Resize((16,16))(att_maps)
        att_maps = att_maps.detach().cpu().numpy()
        r = att_maps.shape[2]
        h = w =  int(np.sqrt(r))
        plt.axis('off')
        fig=plt.figure(0,figsize=(5, 5))
        
        map = att_maps.mean(0)           #[ir]
        map = (map-np.min(map))/(np.max(map)-np.min(map)) * 0.1
        #map[10:14,12:15] = 0 #np.mean(map)
        h = w =  map.shape[0]
        score = S

        image = self.data_transforms(imageOrg.clone())
        image -= image.min()
        image /= image.max()
        if is_top:
            idxs_top=np.argsort(-score)[:n_top_attr]
        else:
            idxs_top=np.argsort(score)[:n_top_attr]

        fig=plt.figure(0,figsize=(5, 5))
        #pdb.set_trace()
        for idx_ctxt,idx_attr in enumerate(idxs_top):
            # ax=plt.subplot(4, 5, idx_ctxt+2)
            plt.imshow(image)
            
            ax = plt.gca()
            alp_curr = map

            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=image_size/h, sigma=5)

            plt.imshow(alp_img, alpha=0.5, cmap=plt.cm.jet)
            plt.axis('off')  # 去坐标轴
            plt.show()

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path,dpi=500)
            plt.close()
    
    def _forward_once_(self, batched_inputs, gt_instances=None, is_attack = False, iter = None):

        images = self.preprocess_image(batched_inputs)
        gt_classes = [x["instances"].gt_classes for x in batched_inputs]
        features = self.backbone(images.tensor)
        # print(features)
        # print(features["res4"].shape)
        # exit()
        """
            - 2024.6.21        
        """
        
        #onlymamba = self.onlymamba
        if self.cfg.MODEL.BACKBONE.WITHMAMBA:     
            B, C, H, W = features["res4"].shape
            features["res4"] = features["res4"].view(B, C, H * W).permute(0, 2, 1)    
            hidden_states, residual = self.mamba(features["res4"])
            features["res4"] = (hidden_states + residual).permute(0, 2, 1).view(B, C, H, W)
        
        
        if self.transform_feat is not None and self.training:
            features = {k: self.transform_feat(v) for k, v in features.items()}
        if self.cfg.MODEL.BACKBONE.FEATURE_AUG_TYPE == "Cutout":
            features = cutout_feat(features, 8)
        elif self.cfg.MODEL.BACKBONE.FEATURE_AUG_TYPE == "Cutmix":
            features, gt_classes = cutmix(features, gt_classes)
        elif self.cfg.MODEL.BACKBONE.FEATURE_AUG_TYPE == "Mixup":
            features = mixup_feature(features)


        if self.WithExtConv:
            for i in range(self.ExtConvLayer):
                for f in features.keys():
                    features[f] = getattr(self, f'ext_conv_{i}')(features[f]) + features[f]
                # features = [(getattr(self, f'ext_conv_{i}')(features[f]) + features[f]) for f in features.keys()]

        if self.cfg.MODEL.BACKBONE.WITHECEA:
            results = [self.lateral_conv_in(features[key]) for key in features.keys()]
            
            #apply trans encoder
            mlvl_feats = tuple(results)
            batch_input_shape = tuple(mlvl_feats[0].shape[-2:])
            input_img_h, input_img_w = batch_input_shape
            batch_size = mlvl_feats[0].size(0)
            img_masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            
            for img_id in range(batch_size):
                img_h, img_w = mlvl_feats[0].shape[-2:]
                img_masks[img_id, :img_h, :img_w] = 0
            
            mlvl_masks = []
            mlvl_positional_encodings = []
            
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))
            query_embeds = None
            for out in results:
                if not torch.isfinite(out).all():
                    a = 1
            feat_flatten = [out.flatten(2) for out in results]     # outs: list( [BN, channel, width, height] )
                                                                # feat_flatten [BN, channel, sum(width*height)]
            spatial_shapes = [out.shape[2:4] for out in results]
            
            mask_flatten = [mask.flatten(1) for mask in mlvl_masks]
            lvl_pos_embed_flatten = []
            
            for lvl, ( mask, pos_embed) in enumerate(
                    zip( mlvl_masks, mlvl_positional_encodings)):

                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)

                
            feat_flatten = torch.cat(feat_flatten, 2)
            if torch.isnan(feat_flatten).any():
                a = 1
            if torch.isinf(feat_flatten).any():
                a = 1
            mask_flatten = torch.cat(mask_flatten, 1)
            
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            
            spatial_shapes = torch.as_tensor( spatial_shapes, dtype=torch.long, device=lvl_pos_embed_flatten.device )
            
            level_start_index = torch.cat((spatial_shapes.new_zeros( (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            valid_ratios = torch.stack( [ self.get_valid_ratio(m) for m in mlvl_masks], 1)
            
            reference_points = self.get_reference_points(spatial_shapes,valid_ratios, device=feat.device)
                
            feat_flatten = feat_flatten.permute(2, 0, 1)  # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
            kwargs = {}
            memory = self.encoder(
                query=feat_flatten,
                key=None,
                value=None,
                query_pos=lvl_pos_embed_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                **kwargs)
            feat_flatten = memory.permute(1, 2, 0)
            
            bn,channel,_ = feat_flatten.shape
            value_list = feat_flatten.split([H_ * W_ for H_, W_ in spatial_shapes],
                                dim=2)

            out_results = [torch.reshape(value_list[i],[bn, channel, spatial_shapes[i][0], spatial_shapes[i][1]] ) for i in range(len(value_list))]
            
            for f, res in zip(features, out_results):
                features[f] = features[f] * 0.6 + self.lateral_conv_out(res) * 0.4


##############################################################################################################

        if self.SCSM:
            results = [self.lateral_conv_in(features[key]) for key in features.keys()]
            
            #apply trans encoder
            mlvl_feats = tuple(results)
            batch_input_shape = tuple(mlvl_feats[0].shape[-2:])
            input_img_h, input_img_w = batch_input_shape
            batch_size = mlvl_feats[0].size(0)
            img_masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            
            for img_id in range(batch_size):
                img_h, img_w = mlvl_feats[0].shape[-2:]
                img_masks[img_id, :img_h, :img_w] = 0
            
            mlvl_masks = []
            mlvl_positional_encodings = []
            
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))
            query_embeds = None
            for out in results:
                if not torch.isfinite(out).all():
                    a = 1
            feat_flatten = [out.flatten(2) for out in results]     # outs: list( [BN, channel, width, height] )
                                                                # feat_flatten [BN, channel, sum(width*height)]
            spatial_shapes = [out.shape[2:4] for out in results]
            
            mask_flatten = [mask.flatten(1) for mask in mlvl_masks]
            lvl_pos_embed_flatten = []
            
            for lvl, ( mask, pos_embed) in enumerate(
                    zip( mlvl_masks, mlvl_positional_encodings)):

                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)

                
            feat_flatten = torch.cat(feat_flatten, 2)
            if torch.isnan(feat_flatten).any():
                a = 1
            if torch.isinf(feat_flatten).any():
                a = 1
            mask_flatten = torch.cat(mask_flatten, 1)
            
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            
            spatial_shapes = torch.as_tensor( spatial_shapes, dtype=torch.long, device=lvl_pos_embed_flatten.device )
            
            level_start_index = torch.cat((spatial_shapes.new_zeros( (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            valid_ratios = torch.stack( [ self.get_valid_ratio(m) for m in mlvl_masks], 1)
            
            reference_points = self.get_reference_points(spatial_shapes,valid_ratios, device=feat.device)
                
            feat_flatten = feat_flatten.permute(2, 0, 1)  # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
            kwargs = {}

            for i in range(1):
                memory = getattr(self, f'encoder_{i}')(
                    query=feat_flatten,
                    key=None,
                    value=None,
                    query_pos=lvl_pos_embed_flatten,
                    query_key_padding_mask=mask_flatten,
                    spatial_shapes=spatial_shapes,
                    reference_points=reference_points,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios
                )

                feat_flatten = memory.permute(1, 0, 2) # (bs, H*W, embed_dims)
                pooled_C = self.mamba_pooled

                if feat_flatten.shape[1] != pooled_C:
                    feat_mamba = F.interpolate(feat_flatten.permute(0, 2, 1), size=pooled_C, mode='nearest')
                else:
                    feat_mamba = feat_flatten.permute(0, 2, 1)

                feat_flatten_mamba, residual = getattr(self, f'mamba_block_{i}')(feat_mamba)
                feat_flatten_mamba = F.interpolate(feat_flatten_mamba, size=feat_flatten.shape[1], mode='nearest').permute(0, 2, 1)
                
                feat_flatten = (feat_flatten + feat_flatten_mamba).permute(1, 0, 2) # (H*W, bs, embed_dims)

                
            feat_flatten = feat_flatten.permute(1, 2, 0)

            bn , channel,_ = feat_flatten.shape
            value_list = feat_flatten.split([H_ * W_ for H_, W_ in spatial_shapes], dim=2)
            out_results = [torch.reshape(value_list[i],[bn, channel, spatial_shapes[i][0], spatial_shapes[i][1]] ) for i in range(len(value_list))]
                
            for f, res in zip(features, out_results):
                features[f] = features[f] * 0.6 + self.lateral_conv_out(res) * 0.4

###################################################################################################################
        """
            - 2024.7.9
        """

        if self.SFASA:
            results = [self.lateral_conv_in(features[key]) for key in features.keys()]
            
            #apply trans encoder
            mlvl_feats = tuple(results)
            batch_input_shape = tuple(mlvl_feats[0].shape[-2:])
            input_img_h, input_img_w = batch_input_shape
            batch_size = mlvl_feats[0].size(0)
            img_masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            
            for img_id in range(batch_size):
                img_h, img_w = mlvl_feats[0].shape[-2:]
                img_masks[img_id, :img_h, :img_w] = 0
            
            mlvl_masks = []
            mlvl_positional_encodings = []
            
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))
            query_embeds = None
            for out in results:
                if not torch.isfinite(out).all():
                    a = 1
            feat_flatten = [out.flatten(2) for out in results]     # outs: list( [BN, channel, width, height] )
                                                                # feat_flatten [BN, channel, sum(width*height)]
            spatial_shapes = [out.shape[2:4] for out in results]
            
            mask_flatten = [mask.flatten(1) for mask in mlvl_masks]
            lvl_pos_embed_flatten = []
            
            for lvl, ( mask, pos_embed) in enumerate(
                    zip( mlvl_masks, mlvl_positional_encodings)):

                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)

                
            feat_flatten = torch.cat(feat_flatten, 2)
            if torch.isnan(feat_flatten).any():
                a = 1
            if torch.isinf(feat_flatten).any():
                a = 1
            mask_flatten = torch.cat(mask_flatten, 1)
            
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            
            spatial_shapes = torch.as_tensor( spatial_shapes, dtype=torch.long, device=lvl_pos_embed_flatten.device )
            
            level_start_index = torch.cat((spatial_shapes.new_zeros( (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            valid_ratios = torch.stack( [ self.get_valid_ratio(m) for m in mlvl_masks], 1)
            
            reference_points = self.get_reference_points(spatial_shapes,valid_ratios, device=feat.device)
                
            feat_flatten = feat_flatten.permute(2, 0, 1)  # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
            kwargs = {}

            for i in range(3):
                memory = getattr(self, f'encoder_{i}')(
                    query=feat_flatten,
                    key=None,
                    value=None,
                    query_pos=lvl_pos_embed_flatten,
                    query_key_padding_mask=mask_flatten,
                    spatial_shapes=spatial_shapes,
                    reference_points=reference_points,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios
                )
                feat_flatten = memory.permute(1, 0, 2) # (bs, H*W, embed_dims)
                feat_flatten = getattr(self, f'transformer_block_{i}')(feat_flatten)
                feat_flatten = feat_flatten.permute(1, 0, 2) # (H*W, bs, embed_dims)
            feat_flatten = feat_flatten.permute(1, 2, 0)

            bn , channel,_ = feat_flatten.shape
            value_list = feat_flatten.split([H_ * W_ for H_, W_ in spatial_shapes], dim=2)
            out_results = [torch.reshape(value_list[i],[bn, channel, spatial_shapes[i][0], spatial_shapes[i][1]] ) for i in range(len(value_list))]
                
            for f, res in zip(features, out_results):
                features[f] = features[f] * 0.6 + self.lateral_conv_out(res) * 0.4
###################################################################################################################


###################################################################################################################
        """
            - 2024.7.15
        """

        if self.SFASENet:
            results = [self.lateral_conv_in(features[key]) for key in features.keys()]
            
            #apply trans encoder
            mlvl_feats = tuple(results)
            batch_input_shape = tuple(mlvl_feats[0].shape[-2:])
            input_img_h, input_img_w = batch_input_shape
            batch_size = mlvl_feats[0].size(0)
            img_masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            
            for img_id in range(batch_size):
                img_h, img_w = mlvl_feats[0].shape[-2:]
                img_masks[img_id, :img_h, :img_w] = 0
            
            mlvl_masks = []
            mlvl_positional_encodings = []
            
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))
            query_embeds = None
            for out in results:
                if not torch.isfinite(out).all():
                    a = 1
            feat_flatten = [out.flatten(2) for out in results]     # outs: list( [BN, channel, width, height] )
                                                                # feat_flatten [BN, channel, sum(width*height)]
            spatial_shapes = [out.shape[2:4] for out in results]
            
            mask_flatten = [mask.flatten(1) for mask in mlvl_masks]
            lvl_pos_embed_flatten = []
            
            for lvl, ( mask, pos_embed) in enumerate(
                    zip( mlvl_masks, mlvl_positional_encodings)):

                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)

                
            feat_flatten = torch.cat(feat_flatten, 2)
            if torch.isnan(feat_flatten).any():
                a = 1
            if torch.isinf(feat_flatten).any():
                a = 1
            mask_flatten = torch.cat(mask_flatten, 1)
            
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            
            spatial_shapes = torch.as_tensor( spatial_shapes, dtype=torch.long, device=lvl_pos_embed_flatten.device )
            
            level_start_index = torch.cat((spatial_shapes.new_zeros( (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            valid_ratios = torch.stack( [ self.get_valid_ratio(m) for m in mlvl_masks], 1)
            
            reference_points = self.get_reference_points(spatial_shapes,valid_ratios, device=feat.device)
                
            feat_flatten = feat_flatten.permute(2, 0, 1)  # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
            kwargs = {}

            for i in range(3):
                memory = getattr(self, f'encoder_{i}')(
                    query=feat_flatten,
                    key=None,
                    value=None,
                    query_pos=lvl_pos_embed_flatten,
                    query_key_padding_mask=mask_flatten,
                    spatial_shapes=spatial_shapes,
                    reference_points=reference_points,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios
                ) # (H*W, bs, embed_dims)
                feat_flatten = memory.permute(1, 2, 0) # (bs, embed_dims, H*W)
                feat_flatten = getattr(self, f'se_block_{i}')(feat_flatten)
                feat_flatten = feat_flatten.permute(2, 0, 1) # (H*W, bs, embed_dims)
            feat_flatten = feat_flatten.permute(1, 2, 0)

            bn , channel,_ = feat_flatten.shape
            value_list = feat_flatten.split([H_ * W_ for H_, W_ in spatial_shapes], dim=2)
            out_results = [torch.reshape(value_list[i],[bn, channel, spatial_shapes[i][0], spatial_shapes[i][1]] ) for i in range(len(value_list))]
                
            for f, res in zip(features, out_results):
                features[f] = features[f] * 0.6 + self.lateral_conv_out(res) * 0.4
###################################################################################################################


        """
            - 2024.6.27
        """
        # PAMmamba = False
        # if PAMmamba:     
        #     B, C, H, W = features["res4"].shape
        #     features["res4"] = features["res4"].view(B, C, H * W).permute(0, 2, 1)    
        #     hidden_states, residual = self.mamba(features["res4"])
        #     features["res4"] = (hidden_states + residual).permute(0, 2, 1).view(B, C, H, W)
        
        
        
        """
            - 2024.6.28
        """
        
        if self.with_adapter:
            AdapterOut = self.adapter(features["res4"])
            features["res4"] = AdapterOut

    
        is_attack = self.cfg.MODEL.BACKBONE.ATTACK
        if is_attack:
            print("attack is succssfull.")
            #calculate the attack grad
            # features_new = {}
            # for key in features.keys():
            #     features_new[key] = features[key]#.detach().requires_grad_(True)
            #     #features_new[key].requires_grad = True


            features_de_rpn = features
            if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
            proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

            features_de_rcnn = features
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
            results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

            total_norm_loss = sum(detector_losses.values()) + sum(proposal_losses.values())
            
            feature_grad = {key:torch.autograd.grad(total_norm_loss, features[key], retain_graph=True)[0] for key in features}

            features_attack = {}
            # std_deviation = np.std(features, axis=0)
            # print(std_deviation)
            p_norm_bound = 150
            for key in features.keys():              
                
                
                x = 0.1 * torch.sign(feature_grad[key])
                p_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
                # x2 = 0.08 * torch.sign(feature_grad[key])
                # p_norm2 = torch.norm(x2.view(x.size(0), -1), p=2, dim=1)
                # print("p_norm is : ",p_norm)
                # print("p_norm2 is : ",p_norm2)

                # needdeduceepslion = torch.any(p_norm > p_norm_bound)
                # if needdeduceepslion:
                #     features_attack[key] = features[key] + 0.08 * torch.sign(feature_grad[key])
                # else:
                #     features_attack[key] = features[key] + 0.1 * torch.sign(feature_grad[key])

                features_attack[key] = features[key] + 0.1 * torch.sign(feature_grad[key])
                
                if self.cfg.SAVE_VIS_PLT:
                    
                    # needdeduceepslion = torch.any(p_norm > p_norm_bound)
                    # if needdeduceepslion:
                    #     features_attack[key] = features[key] + 0.5 * torch.sign(feature_grad[key])
                    # else:
                    #     features_attack[key] = features[key] + 0.5 * torch.sign(feature_grad[key])
                    
                    orignal_feature = torch.mean(features[key].detach(), dim=1).permute(1,2,0)
                    attack_feature = torch.mean(features_attack[key].detach(), dim=1).permute(1,2,0)
                    for idx in range(orignal_feature.shape[2]):     
                        plt.subplot(1,2,1)
                        plt.imshow(orignal_feature[:,:,idx].cpu().detach(), cmap='viridis')
                        plt.title("Original Feature")
                        plt.axis("off")
                        plt.subplot(1,2,2)
                        plt.imshow(attack_feature[:,:,idx].cpu().detach(), cmap='viridis')
                        plt.title("Adversarial Feature")
                        plt.axis("off")
                        # plt.show()
                        file_name = batched_inputs[idx]['file_name'].split('/')[-1].split('.')[0]
                        img_save_path = os.path.join(self.cfg.OUTPUT_DIR, file_name +'_' + str(key) + '.png')
                        plt.savefig(img_save_path,dpi=500)
                        plt.close()
          

            #plt.imshow(features_attack)
            features_attack_rpn = features_attack            
            if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                features_attack_rpn = {k: self.affine_rpn(decouple_layer(features_attack[k], scale)) for k in features_attack}
            proposals, proposal_attack_losses = self.proposal_generator(images, features_attack_rpn, gt_instances)

            features_attack_rcnn = features_attack
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_attack_rcnn = {k: self.affine_rcnn(decouple_layer(features_attack[k], scale)) for k in features_attack}
            results, detector_attack_losses = self.roi_heads(images, features_attack_rcnn, proposals, gt_instances)

            proposal_losses = {key: 0.8 * proposal_losses[key] + 0.2 * proposal_attack_losses[key] for key in proposal_losses}
            detector_losses = {key: 0.8 * detector_losses[key] + 0.2 * detector_attack_losses[key] for key in detector_losses}
        
        else:
            features_de_rpn = features        
            #print("attack is unsuccssfull.")
            if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
            proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

            features_de_rcnn = features
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
            results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)
            
            # for idx, x in enumerate(batched_inputs):
            #     x['image_id'] = str(idx)
        if self.cfg.SAVE_ATTENTION_MAP:
            for idx, x in enumerate(batched_inputs):
                
                image = x['image']
                S = np.random.rand(1, 1)
                n_top_attr = 1
                attr_name = np.array(['attesntion'])
                attr = np.random.rand(1, 1)
                filenames = batched_inputs[idx]['file_name'].split('/')
                attension_save_path = os.path.join(self.cfg.OUTPUT_DIR, 'attention')
                os.mkdir(attension_save_path) if not os.path.exists(attension_save_path) else None
                save_path = os.path.join(attension_save_path, filenames[-1])
                #att_maps = np.zeros((1, 1, 7*7)) + 0.1
                att_maps = features[self.roi_heads.in_features[-1]][idx]
                self.visualize_attention(images[0], att_maps, S, n_top_attr, attr_name, attr,save_path, is_top=True)
                
            #self.features = features
        if not self.training and iter is not None and self.cfg.SAVE_ROI_FEATURE:
            gt_roi_features = []
            #get the roi feature of batched_inputs
            gt_boxes = [input['instances'].gt_boxes.to('cuda') for input in batched_inputs]
            gt_labels = [input['instances'].gt_classes.detach().cpu() for input in batched_inputs]
            gt_labels = np.array([x.detach().cpu().numpy() for x in gt_labels])
            if gt_labels.shape[1] > 0:
                gt_roi_features = self.roi_heads._shared_roi_transform( [features[f] for f in features.keys()], gt_boxes )
                gt_roi_features = torch.mean(gt_roi_features.view(gt_roi_features.size(0), gt_roi_features.size(1), -1), dim=2)
                gt_roi_features = gt_roi_features.detach().cpu().numpy()

                iter_name = "{0}".format(iter * torch.cuda.device_count() + torch.cuda.current_device())
            
                np_save_path = os.path.join(self.cfg.OUTPUT_DIR, 'features')
                os.mkdir(np_save_path) if not os.path.exists(np_save_path) else None

                np.save(os.path.join(np_save_path, iter_name + '_gt_roi_features.npy'), gt_roi_features)
                np.save(os.path.join(np_save_path, iter_name + '_gt_labels.npy'), gt_labels)
        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        #resize image to 224 X 224
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        # new_size = 224
        # images = [x.float()/255.0 for x in images]
        # images = [F.interpolate(x.unsqueeze(0), size=(new_size,new_size), mode='bilinear').squeeze(0) for x in images]
        # images = [(x * 255.0).to(torch.uint8) for x in images]

        # images = [x["image"].to(self.device) for x in batched_inputs]
        gt_classes = [x["instances"].gt_classes for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        #apply CutMix to images
        
        if self.transform is not None and self.training:
            images = [self.transform(x) for x in images]
        if self.cfg.MODEL.BACKBONE.DATA_AUG_TYPE == "CutMix":
            images, targets = cutmix(images, gt_classes)
        elif self.cfg.MODEL.BACKBONE.DATA_AUG_TYPE == "Cutout":
            images = cutout(images, 16)
        elif self.cfg.MODEL.BACKBONE.DATA_AUG_TYPE == "Mixup":
            images = mixup(images)

        # images = [self.transform(x) for x in images]
        if hasattr(self.backbone, "size_divisibility"):
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        else:
            images = ImageList.from_tensors(images, 32)
        
        return images
    def get_features(self):
        return self.features

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        if spatial_shapes.shape[0] == 1:
            lvl = 0
            H, W = spatial_shapes[0]
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        else:
            
            for lvl, (H, W) in enumerate(spatial_shapes):
                #  TODO  check this 0.5
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, H - 0.5, H, dtype=torch.float32, device=device),
                    torch.linspace(
                        0.5, W - 0.5, W, dtype=torch.float32, device=device))
                ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
                ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            
        
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

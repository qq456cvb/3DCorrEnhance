import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import cv2
import matplotlib.cm as cm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import functional
from visdom import Visdom

from data_utils.dataset import AugmentedDataset, ObjaverseCorrDataset
from utils.functions import fix_random_seeds, sigmoid, interpolate_features
from utils.model import _LoRA_qkv


class FinetuneDINO(pl.LightningModule):
    def __init__(self, r, backbone_size, reg=False, datasets=None):
        super().__init__()
        assert r > 0
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        if reg:
            self.backbone_arch = f"{self.backbone_arch}_reg"
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        self.datasets = datasets
        
        self.lora_layer = list(range(len(dinov2.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze first
        for param in dinov2.parameters():
            param.requires_grad = False

        # finetune the last 4 blocks
        for t_layer_i, blk in enumerate(dinov2.blocks[-4:]):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()

        self.dinov2 = dinov2
        self.downsample_factor = 8

        self.refine_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        self.thresh3d_pos = 5e-3
        self.thres3d_neg = 0.1
        
        self.patch_size = 14
        self.target_res = 640
        
        self.input_transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        checkpoint['state_dict'] = {
            'refine_conv': self.refine_conv.state_dict(),
        }
        checkpoint.update(a_tensors)
        checkpoint.update(b_tensors)
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # print(checkpoint.keys())
        self.refine_conv.load_state_dict(checkpoint['state_dict']['refine_conv'])
        
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)
        self.loaded = True
            
    def train_dataloader(self):
        self.datasets = [ObjaverseCorrDataset('data/objaverse_renderings', 10_000)]        
        dataset = AugmentedDataset(ConcatDataset(self.datasets))
        g = torch.Generator()
        g.manual_seed(42)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            # worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id + int(datetime.now().timestamp())),
            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id),
            generator=g,
        )
    
    def get_nearest(self, query, database):
        dist = torch.cdist(query, database)
        min_dist, min_idx = torch.min(dist, -1)
        return min_dist, min_idx
    
    def get_feature(self, rgbs, pts, normalize=True):
        tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
        if rgbs.shape[-2] > rgbs.shape[-1]:
            tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
        patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
        rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
        resize_factor = [(patch_w * self.patch_size) / rgbs.shape[-1], (patch_h * self.patch_size) / rgbs.shape[-2]]
        
        pts = pts * torch.tensor(resize_factor).to(pts.device)
        
        result = self.dinov2.forward_features(self.input_transform(rgb_resized))
        
        feature = result['x_norm_patchtokens'].reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2)
        feature = self.refine_conv(feature)
            
        feature = interpolate_features(feature, pts, h=patch_h * 14, w=patch_w * 14, normalize=False).permute(0, 2, 1)
        if normalize:
            feature = F.normalize(feature, p=2, dim=-1)
        return feature
    
    def get_feature_wo_kp(self, rgbs, normalize=True):
        tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
        if rgbs.shape[-2] > rgbs.shape[-1]:
            tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
        patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
        rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
        result = self.dinov2.forward_features(self.input_transform(rgb_resized))
        feature = result['x_norm_patchtokens'].reshape(rgbs.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2)
        feature = self.refine_conv(feature)
        feature = functional.resize(feature, (rgbs.shape[-2], rgbs.shape[-1])).permute(0, 2, 3, 1)
        if normalize:
            feature = F.normalize(feature, p=2, dim=-1)
        return feature
        
    def training_step(self, batch, batch_idx):
        # print(batch['obj_name_1'])
        rgb_1, pts2d_1, pts3d_1 = batch['rgb_1'], batch['pts2d_1'], batch['pts3d_1']
        rgb_2, pts2d_2, pts3d_2 = batch['rgb_2'], batch['pts2d_2'], batch['pts3d_2']
        
        desc_1 = self.get_feature(rgb_1, pts2d_1, normalize=True)
        desc_2 = self.get_feature(rgb_2, pts2d_2, normalize=True)
        
        kp3d_dist = torch.cdist(pts3d_1, pts3d_2)  # B x S x T
        sim = torch.bmm(desc_1, desc_2.transpose(-1, -2))  # B x S x T
        
        pos_idxs = torch.nonzero(kp3d_dist < self.thresh3d_pos, as_tuple=False)
        pos_sim = sim[pos_idxs[:, 0], pos_idxs[:, 1], pos_idxs[:, 2]]
        rpos = sigmoid(pos_sim - 1., temp=0.01) + 1  # si = 1  # pos
        neg_mask = kp3d_dist[pos_idxs[:, 0], pos_idxs[:, 1]] > self.thres3d_neg # pos x T
        rall = rpos + torch.sum(sigmoid(sim[pos_idxs[:, 0], pos_idxs[:, 1]] - 1., temp=0.01) * neg_mask.float(), -1)  # pos
        ap1 = rpos / rall
        
        # change teh order
        rpos = sigmoid(1. - pos_sim, temp=0.01) + 1  # si = 1  # pos
        neg_mask = kp3d_dist[pos_idxs[:, 0], pos_idxs[:, 1]] > self.thres3d_neg # pos x T
        rall = rpos + torch.sum(sigmoid(sim[pos_idxs[:, 0], pos_idxs[:, 1]] - pos_sim[:, None].repeat(1, sim.shape[-1]), temp=0.01) * neg_mask.float(), -1)  # pos
        ap2 = rpos / rall
        
        ap = (ap1 + ap2) / 2
        
        loss = torch.mean(1. - ap)
        
        self.log('loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW([layer.weight for layer in self.w_As]
                                 + [layer.weight for layer in self.w_Bs]
                                 + list(self.refine_conv.parameters()), lr=1e-5, weight_decay=1e-4)


from pytorch_lightning.callbacks import ModelCheckpoint
import hydra

@hydra.main(config_path='./config', config_name='finetune', version_base='1.2')
def main(cfg):
    fix_random_seeds()
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    
    logger = TensorBoardLogger(output_dir)
    pl_module = FinetuneDINO(r=4, backbone_size=cfg.backbone, reg=cfg.reg)
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', callbacks=[ModelCheckpoint(save_last=True, every_n_epochs=10, save_top_k=-1)], 
                         gradient_clip_val=1.0, logger=logger)
    trainer.fit(pl_module)
    
if __name__ == '__main__':
    main()
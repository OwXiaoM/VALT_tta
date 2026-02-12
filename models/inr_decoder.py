import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.siren import Siren
from utils import embed2affine
from scipy.ndimage import label
import scipy.ndimage as ndi

class INR_Decoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        args_inr = args['inr_decoder']
        self.device = device
        self.sr_dims = sum(args_inr['out_dim'][:-1])
        self.out_dim = sum(args_inr['out_dim'])
        self.mod_dim = args_inr['hidden_size'] * len(args_inr['modulated_layers']) * 2
        self.modulator = Modulator(args_inr['latent_dim'], kernel_size=args_inr['cnn_kernel_size'])
        
        # [新增] 获取 MoE 相关配置，默认为 False/默认值以保持兼容性
        use_moe = args_inr.get('use_moe', False)
        num_experts = args_inr.get('num_experts', 4)
        moe_k = args_inr.get('moe_k', 2)

        # [修改] 将 MoE 参数传递给 Siren
        self.sr_net = Siren(
            in_size=args_inr['in_dim'], 
            lat_size=args_inr['latent_dim'][0] + args_inr['cond_dims'], 
            out_size=self.out_dim, 
            hidden_size=args_inr['hidden_size'],
            num_layers=args_inr['num_hidden_layers'], 
            f_om=args_inr['omega'][0], 
            h_om=args_inr['omega'][1],
            outermost_linear=True, 
            modulated_layers=args_inr['modulated_layers'],
            # 新增参数
            use_moe=use_moe,
            num_experts=num_experts,
            moe_k=moe_k
        )

    def forward(self, coords, latent_vecs, condition_vecs, tfs=None, idcs_df=None):
        """
        Args:
            coords: (N, 3)
            latent_vecs: (N, l_channels, l_x, l_y, l_z)
            condition_vecs: (N, c_channels)
            tfs: (N, 6)
        Returns:
            output: (N, out_dim)
            aux_loss: scalar tensor (MoE load balancing loss)
        """
        coords = self.transform(coords, tfs) if tfs is not None else coords
        modulations = self.modulator(latent_vecs)[idcs_df]
        modulations_interp = self.spatial_interpolation(coords, modulations, condition_vecs)
        
        # 执行前向传播
        output = self.sr_net((coords, modulations_interp))
        
        # [新增] 获取 MoE 的辅助损失 (Load Balancing Loss)
        # 即使 use_moe=False，Siren.get_moe_loss 也应设计为返回 0
        aux_loss = self.sr_net.get_moe_loss()
        
        return output, aux_loss

    def inference(self, coords, latent_vec, condition_vec, img_shape, tfs=None, step_size=100000):
        """
        Inference of the INR decoder for volume generation.
        """
        # 1. 初始化输出容器
        output = torch.empty((coords.shape[0], self.out_dim)).to(device=self.device)
        
        # 2. 坐标变换
        coords = self.transform(coords, tfs.expand(coords.shape[0], -1)) if tfs is not None else coords
        
        # 3. 分块推理 (Chunking)
        for i in range(0, coords.shape[0], step_size):
            c = coords[i:i+step_size]
            
            # 确保索引在 GPU 上
            idcs_df = torch.zeros(c.shape[0], dtype=torch.long, device=self.device)
            
            cv = condition_vec.expand(c.shape[0], -1)
            
            # [修改] forward 现在返回 (output, aux_loss)，我们只需要 output
            # 这里的 aux_loss 在推理阶段通常会被忽略
            batch_out, _ = self.forward(c, latent_vec, cv, idcs_df=idcs_df)
            
            output[i:i + step_size] = batch_out

        raw_imgs = output[..., :self.sr_dims]
        
        # 4. 归一化逻辑 (针对 MRA 等稀疏数据优化)
        imgs = torch.clamp(raw_imgs, 0, 1)
        
        # 5. Reshape
        imgs = imgs.reshape(img_shape + [-1])

        # 6. 处理分割结果
        seg_hard = torch.argmax(output[..., self.sr_dims:], dim=-1).reshape(img_shape + [-1])
        seg_soft = torch.nn.functional.softmax(output[..., self.sr_dims:], dim=-1).reshape(img_shape + [-1])
       
        # 7. 合并通道
        modalities_rec = torch.cat((imgs, seg_hard, seg_soft), dim=-1)

        if self.args['mask_reconstruction']:
            return self.mask_reconstruction(modalities_rec, seg_hard)
        else:
            return modalities_rec

    @staticmethod
    def transform(coords, tfs, inverse=False):
        R, t = embed2affine(tfs)
        if inverse:
            R = R.inverse()
            t = -torch.einsum('nij,nj->ni', R, t)
        coords = torch.einsum('nxy,ny->nx', R, coords) + t
        return coords

    @staticmethod
    def spatial_interpolation(coords, latents, condition_vecs=None):
        """
        Spatial interpolation of the latent vector: 4D to 1D
        coords: (N, 4)
        latents: (N, C, l_x, l_y, l_z)
        """
        coords = coords[:, None, None, None, :] # (N, 1, 1, 1, 3)
        latents = F.grid_sample(latents, coords, mode='bilinear', align_corners=True, 
                                padding_mode='border').squeeze()
        if condition_vecs is not None:
            latents = torch.concat((latents, condition_vecs), dim=-1)
        return latents
        
    def mask_reconstruction(self, recs, seg):
        mask = self.connected_components(seg)
        mask = mask.expand_as(recs)
        return recs * mask

    def connected_components(self, seg, bg_label_str='Background'):
        bg_label = self.args['dataset']['label_names'].index(bg_label_str)
        mask = ((seg > 0) & (seg != bg_label)).detach().cpu().numpy()
        shp = np.array(list(mask.shape[:-1]))
        ps = ((shp * 0.1) //2).astype(int)  # patch size 10% of image size
        cp = shp // 2
        # get connected components
        labeled_mask, num_labels = label(mask)
        # get majority label of center patch
        center_label = labeled_mask[cp[0]-ps[0]:cp[0]+ps[0], cp[1]-ps[1]:cp[1]+ps[1], cp[2]-ps[2]:cp[2]+ps[2]].flatten()
        majority_label = np.argmax(np.bincount(center_label))
        # set all other labels to 0
        mask = mask * (labeled_mask == majority_label)
        mask_blur = (ndi.gaussian_filter(mask.astype(np.float32), sigma=1.0) > 0.001).astype(np.uint8)

        return torch.from_numpy(mask_blur).to(self.device, torch.float)

class Modulator(nn.Module):
    """
    Modulater for the latent vector based on CNNs. 
    Args:
        latent_dims: (C_latent, l_x, l_y, l_z)
    """
    def __init__(self, latent_dims, kernel_size=3):
        super().__init__()
        if kernel_size > 0:
            self.conv = nn.Conv3d(latent_dims[0], latent_dims[0], kernel_size, padding='same')
        else:
            self.conv = nn.Identity()

    def forward(self, latent_vecs):
        return self.conv(latent_vecs)


class LatentRegressor(nn.Module):
    """
    Latent regressor to regress information from the latent vector, e.g., birth_age.
    """
    def __init__(self, latent_dims):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dims[0], 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, latent_vecs):
        return self.sequence(latent_vecs)
    
import wandb as wd
import pandas as pd
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# 引用原有的模型和数据定义
from models.inr_decoder import INR_Decoder, LatentRegressor
from data_loading.dataset import Data
from utils import *

class AtlasBuilderDDP:
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        self.rank = args['rank']
        self.world_size = args['world_size']
        
        self.loss_criterion = Criterion(args).to(self.device)
        self._init_atlas_training()
        self.train_on_data()

    def train_on_data(self):
        # 0. 初始同步：确保所有 GPU 的 Latents 初始值一致
        # 虽然 seed 固定通常能保证一致，但保险起见做一次广播
        self.broadcast_latents_from_rank0()

        if len(self.args['load_model']['path']) > 0 and self.rank == 0: 
            self.validate(epoch_train=0) 
        
        # 确保 Rank 0 验证完再继续
        dist.barrier()
            
        loss_hist_epochs = []
        start_time = time.time()
        total_epochs = self.args['epochs']['train']
        
        for epoch in range(total_epochs):
            if 'train' in self.dataloaders and hasattr(self.dataloaders['train'], 'sampler'):
                self.dataloaders['train'].sampler.set_epoch(epoch)

            # 注意：在 DDP 中，re_init_latents 非常危险，除非你立刻做同步
            # 建议暂时设为 False
            if self.args['optimizer']['re_init_latents']: 
                self.re_init_latents()
                
            loss = self.train_epoch(epoch, split='train')
            loss_hist_epochs.append(loss)
            
            if self.rank == 0:
                print(f"Training: Epoch: {epoch}, Loss: {np.mean(loss_hist_epochs):.4f}, Total Time Epoch: {time.time() - start_time:.2f}s")

            # 验证与保存
            if epoch > 0 and (epoch % self.args['validate_every'] == 0 or epoch == total_epochs - 1):
                # [核心修正] 在验证/保存前，必须把分散在各 GPU 的 Latents 收集起来
                self.synchronize_latents(split='train')
                
                if self.rank == 0:
                    self.validate(epoch)
                
                dist.barrier() # 等待 Rank 0 完成验证
                
            self._update_scheduler(split='train')
            
        return np.mean(loss_hist_epochs)

    def train_epoch(self, epoch, split):
        self.inr_decoder[split].train() if split == 'train' else self.inr_decoder[split].eval()
        loss_hist_batches = []
        
        for i, batch in enumerate(self.dataloaders[split]):
            # DDP 加载数据可能会慢，只在 Rank 0 打印一次作为参考
            if self.rank == 0 and i == 0:
                pass 
            
            loss = self.train_batch(batch, epoch, split)
            loss_hist_batches.append(loss)
            
            if self.rank == 0 and (i % 50 == 0): # 减少打印频率
                 print(f"Split: {split}, Epoch: {epoch}, Batch: {i}/{len(self.dataloaders[split])}, Loss: {loss:.4f}")
                 
        return np.mean(loss_hist_batches)
        
    def train_batch(self, batch, epoch, split='train'):
        loss_hist_samples = []
        n_smpls = self.args['n_samples']
        seg_weight = self.args['optimizer']['seg_weight'] if split == 'train' else 0.0
        coords_batch, values_batch, conditions_batch, idx_df_batch = to_device(batch, self.device)
        sample_iterator = range(0, idx_df_batch.shape[0], n_smpls)
        
        for i, smpls in enumerate(sample_iterator):
            self.optimizers[split].zero_grad()
            coords = coords_batch[smpls:smpls + n_smpls]
            values = values_batch[smpls:smpls + n_smpls]
            idx_df = idx_df_batch[smpls:smpls + n_smpls].squeeze()
            conditions = conditions_batch[smpls:smpls + n_smpls] if split == 'train' else self.conditions_val[idx_df]

            with torch.autocast(device_type='cuda', enabled=self.args['amp']):
                # idcs_df 传入，确保拿到正确的 transformation/latent
                values_p, aux_loss = self.inr_decoder[split](coords, self.latents[split], conditions,
                                            self.transformations[split][idx_df], idcs_df=idx_df)
                
                loss = self.loss_criterion(values_p, values, self.transformations[split][idx_df], 
                                           moe_loss=aux_loss, seg_weight=seg_weight)

            if self.args['amp']:    
                self.grad_scalers[split].scale(loss['total']).backward()
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                loss['total'].backward()
                self.optimizers[split].step()

            loss_hist_samples.append(loss['total'].item())
            
            if (i % 100 == 0 or i == (len(sample_iterator) - 1)) and self.rank == 0:
                log_loss(loss, epoch, split, self.args['logging'])

        return np.mean(loss_hist_samples)

    # ================= [核心 DDP 同步逻辑] =================

    def synchronize_latents(self, split='train'):
        """
        将所有 GPU 上训练的 Latent 和 Transformation 汇总。
        逻辑：创建一个全零 Tensor，每个 Rank 只填入自己负责的那部分 Index 的数据，
        然后通过 All_Reduce(SUM) 得到完整的 Tensor。
        """
        # 1. 计算当前 Rank 负责的 Indices (必须与 DistributedSampler 逻辑一致)
        total_len = len(self.datasets[split])
        indices = torch.arange(total_len).to(self.device)
        
        # DistributedSampler 默认分配策略: indices[rank::world_size]
        my_indices = indices[self.rank::self.world_size]
        
        # 2. 同步 Latents
        # 创建临时全零容器
        temp_latents = torch.zeros_like(self.latents[split].data)
        
        # 填入本机训练好的数据 (其他位置保持为 0)
        temp_latents[my_indices] = self.latents[split].data[my_indices]
        
        # 归约求和：因为其他机器对应位置是 0，SUM 操作就等于 Gather
        dist.all_reduce(temp_latents, op=dist.ReduceOp.SUM)
        
        # 更新本地 Latents (现在大家都有完整的最新数据了)
        self.latents[split].data.copy_(temp_latents)
        
        # 3. 同步 Transformations (如果有)
        if self.args['inr_decoder']['tf_dim'] > 0:
            temp_tfs = torch.zeros_like(self.transformations[split].data)
            temp_tfs[my_indices] = self.transformations[split].data[my_indices]
            dist.all_reduce(temp_tfs, op=dist.ReduceOp.SUM)
            self.transformations[split].data.copy_(temp_tfs)
            
        if self.rank == 0:
            print(f"[DDP] Latents and Transformations synchronized for {split}.")

    def broadcast_latents_from_rank0(self):
        """初始化时使用，确保所有人起点一致"""
        dist.broadcast(self.latents['train'].data, src=0)
        if self.args['inr_decoder']['tf_dim'] > 0:
            dist.broadcast(self.transformations['train'].data, src=0)

    # ========================================================
  
    def validate(self, epoch_train):
        # 此时 self.latents['train'] 已经是同步过的完整数据，Rank 0 可以安全使用
        self.save_state(epoch_train)
        if self.args['generate_cond_atlas']: 
            self.generate_atlas(epoch_train, n_max=100)

        print(f"Starting inference for Epoch {epoch_train}...")
        
        # 采样部分训练集 Subject 做验证
        num_train = len(self.datasets['train'])
        train_indices = [0, 2, 3] if num_train > 3 else list(range(num_train))
        metrics_train = self.generate_subjects_from_df(idcs_df=train_indices, epoch=epoch_train, split='train')
        log_metrics(self.args, metrics_train, epoch_train, df=self.datasets['train'].df, split='train')

        self._init_validation() 
        
        # Val 阶段也是同样逻辑，只有 Rank 0 跑
        # 如果需要在 Val 阶段做 Test-Time Optimization (TTO)，
        # 必须确保 Val 阶段也用 DDP 或者只用单卡跑 Val。
        # 鉴于 Val 数据量通常小，这里让 Rank 0 独占跑 Val 比较简单
        
        for epoch_val in range(self.args['epochs']['val']):
            # 注意：这里的 train_epoch 调用的是 split='val'
            # 因为 _init_validation 中没有用 DDP Sampler，所以其实是单卡跑
            self.train_epoch(epoch=epoch_val, split='val') 
            self._update_scheduler(split='val') 
            self.analyze_latent_space(epoch_train, epoch_val=epoch_val)
            
        metrics_val = self.generate_subjects_from_df(idcs_df=range(len(self.datasets['val'])), 
                                                    epoch=epoch_val, split='val')
        log_metrics(self.args, metrics_val, epoch_train, df=self.datasets['val'].df, split='val')

    def _init_inr(self, state_dict=None, split='train'):
        self.args['inr_decoder']['cond_dims'] = sum([self.args['dataset']['conditions'][c] 
                                                     for c in self.args['dataset']['conditions']])
        
        model = INR_Decoder(self.args, self.device).to(self.device)
        
        if state_dict is not None:
            # 去除 DDP 带来的 module. 前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)

        if split == 'train':
            # 训练时使用 DDP 包裹
            self.inr_decoder[split] = DDP(model, device_ids=[self.args['local_rank']], output_device=self.args['local_rank'], find_unused_parameters=True)
        else:
            # 验证时不使用 DDP (因为只在 Rank 0 跑)
            self.inr_decoder[split] = model

    def _init_dataloading(self, tsv_file=None, df_loaded=None, split='train'):
        tsv_file = pd.read_csv(self.args['dataset']['tsv_file'], sep='\t') if tsv_file is None else tsv_file
        self.datasets[split] = Data(self.args, tsv_file, split=split, df_loaded=df_loaded)

        sampler = None
        shuffle = (split == 'train')
        
        if split == 'train':
            # [关键] shuffle=False!
            # 必须保证每个 Rank 固定负责特定的 Subject，否则 Latent 训练不收敛。
            # 如果设为 True，Epoch 1 Rank 0 训练 Subj A，Epoch 2 Rank 1 训练 Subj A (但是是从初始噪声开始)，
            # 这样 Subj A 永远练不好。
            sampler = DistributedSampler(self.datasets[split], 
                                         num_replicas=self.world_size, 
                                         rank=self.rank, 
                                         shuffle=False) 
            shuffle = False # 使用 sampler 时 loader shuffle 必须关

        self.dataloaders[split] = DataLoader(
            self.datasets[split], 
            batch_size=self.args['batch_size'], 
            num_workers=8, # 适当增加
            shuffle=shuffle, 
            sampler=sampler,
            collate_fn=self.datasets[split].collate_fn, 
            pin_memory=True,
            persistent_workers=True, 
            prefetch_factor=2
        )

        if self.rank == 0:
            print(f"Initialized DDP dataloader for {split} with {len(self.datasets[split])} subjects.")

    def save_state(self, epoch, split='train'):
        if self.rank != 0: return # 只有 Rank 0 保存

        if self.args['save_model']:
            log_dir = self.args['output_dir']
            
            # 获取原始模型 (Unwrap DDP)
            model_to_save = self.inr_decoder[split]
            if isinstance(model_to_save, DDP):
                model_to_save = model_to_save.module
                
            torch.save({
                'epoch': epoch,
                # 此时 self.latents 已经通过 synchronize_latents 同步过，是完整的
                'latents': self.latents[split].cpu(),
                'transformations': self.transformations[split].cpu(),
                'inr_decoder': model_to_save.state_dict(),
                'tsv_file': self.datasets[split].tsv_file,
                'dataset_df': self.datasets[split].df,
                'args': self.args
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Saved model state to {os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")}')

    # --- 以下辅助函数只需适配 module 调用即可 ---

    def generate_subject_from_latent(self, latent_vec, condition_vector, transformation=None, split='train'):
        grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
        model = self.inr_decoder[split]
        if isinstance(model, DDP): model = model.module
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=self.args['amp']):
                volume_inf = model.inference(grid_coords, latent_vec, condition_vector, 
                                                        grid_shape, transformation)
        return volume_inf

    def generate_subjects_from_df(self, idcs_df=None, epoch=0, split='train'):
        import nibabel as nib 
        metrics = []
        model = self.inr_decoder[split]
        if isinstance(model, DDP): model = model.module

        # 辅助函数: generate_native_grid 保持不变... (省略以节省篇幅，直接用原有的)
        def generate_native_grid(header_nii, world_bbox):
            # ... (复制原有的代码逻辑) ...
            shape = header_nii.shape
            affine = header_nii.affine
            i = torch.arange(0, shape[0], device=self.device)
            j = torch.arange(0, shape[1], device=self.device)
            k = torch.arange(0, shape[2], device=self.device)
            grid = torch.meshgrid(i, j, k, indexing='ij')
            grid_coords_idx = torch.stack(grid, dim=-1).reshape(-1, 3).float()
            affine_torch = torch.tensor(affine, dtype=torch.float32, device=self.device)
            ones = torch.ones((grid_coords_idx.shape[0], 1), device=self.device)
            grid_coords_homo = torch.cat([grid_coords_idx, ones], dim=1)
            grid_coords_phys = (affine_torch @ grid_coords_homo.T).T[:, :3]
            img_center_index = torch.tensor(shape, device=self.device) / 2.0
            center_homo = torch.cat([img_center_index, torch.tensor([1.0], device=self.device)])
            geometric_center = (affine_torch @ center_homo)[:3]
            grid_coords_norm = grid_coords_phys - geometric_center
            wb_torch = torch.tensor(world_bbox, dtype=torch.float32, device=self.device)
            grid_coords_norm = grid_coords_norm / (wb_torch / 2.0)
            return grid_coords_norm, list(shape), affine

        for idx_df in idcs_df:
            df_row_dict = self.datasets[split].df.iloc[idx_df].to_dict()
            ref_mod_path = df_row_dict[self.args['dataset']['modalities'][0]]
            ref_nii = nib.load(ref_mod_path)
            grid_coords, grid_shape, affine = generate_native_grid(ref_nii, self.args['dataset']['world_bbox'])
            
            with torch.no_grad():
                transformations = self.transformations[split][idx_df, None]
                conditions = self.datasets[split].load_conditions(df_row_dict).to(self.device)
                
                volume_inf = model.inference(
                    grid_coords, 
                    self.latents[split][idx_df:idx_df+1], 
                    conditions, 
                    grid_shape, 
                    transformations
                )
            
            if self.args['compute_metrics']:
                metrics.append(compute_metrics(self.args, volume_inf, affine, df_row_dict, epoch, split))
            elif self.args['save_imgs'][split]:
                save_subject(self.args, volume_inf, affine, df_row_dict, epoch, split)
        
        return metrics

    def generate_atlas(self, epoch=0, n_max=100):
        # 同样使用 unwrap 后的 model
        model = self.inr_decoder['train']
        if isinstance(model, DDP): model = model.module
        model.eval()
        
        # ... (其余逻辑与原 build_atlas.py 保持完全一致，只需替换 self.inr_decoder['train'] 为 model) ...
        print(f"Generating atlases...\n")
        grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
        temp_steps = self.args['atlas_gen']['temporal_values']
        atlas_list = []
        
        with torch.no_grad():
            for temp_step in temp_steps:
                temp_step_normed = normalize_condition(self.args, 'scan_age', temp_step)
                mean_latent = self.get_mean_latent('scan_age', temp_step_normed, n_max=n_max)
                condition_vectors = generate_combinations(self.args, self.args['atlas_gen']['conditions'])
                cond_list = []
                for c_v in condition_vectors:
                    if self.args['dataset']['conditions'].get('scan_age', False):
                        c_v = [temp_step_normed] + c_v
                    c_v = torch.tensor(c_v, dtype=torch.float32).to(self.device)
                    values_p = model.inference(grid_coords, mean_latent, c_v, grid_shape, None)
                    seg = values_p[:, :, :, -1]
                    seg[seg==4] = 0
                    values_p[:, :, :, -1] = seg
                    cond_list.append(values_p.detach().cpu())
                    torch.cuda.empty_cache()
                atlas_list.append(torch.stack(cond_list, dim=-1))
        atlas_list = torch.stack(atlas_list, dim=-1) 
        save_atlas(self.args, atlas_list, affine, temp_steps, condition_vectors, epoch=epoch)

    # 包含所有其他辅助函数 (get_mean_latent, analyze_latent_space, load_checkpoint 等)
    # 这些函数逻辑基本不变，只需确保涉及模型调用时处理 DDP wrapping
    
    def get_mean_latent(self, condition_key, condition_mean, n_max=100, split='train'):
        # 逻辑不变
        c_ratio = 2 / (self.args['dataset']['constraints'][condition_key]['max'] - self.args['dataset']['constraints'][condition_key]['min'])
        span_weeks = self.args['atlas_gen']['gaussian_span']
        sigma = 0.5 * span_weeks * c_ratio * self.args['atlas_gen']['cond_scale']
        latents = self.latents[split]
        condition_values, df_idcs = self.datasets[split].get_condition_values(condition_key, normed=True, device=self.device)
        weights = torch.exp(-(condition_values - condition_mean)**2 / (2*(sigma**2)))
        n_max = min(n_max, len(weights))
        weights[torch.argsort(weights, descending=True)[n_max:]] = 0
        weights = weights / torch.sum(weights)
        weights = weights[:, None, None, None, None] 
        mean_latent = torch.sum(latents * weights, dim=0, keepdim=True)
        return mean_latent

    def analyze_latent_space(self, epoch, epoch_val=0):
        # 略，与原版一致
        pass

    def load_checkpoint(self, chkp_path=None, epoch=None):  
        chkp_path = os.path.join(chkp_path, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f'State file {chkp_path} not found!')
        chkp = torch.load(chkp_path, weights_only=False)
        self._init_dataloading(chkp['tsv_file'], chkp['dataset_df'])
        self._init_inr(chkp['inr_decoder'], split='train')
        self._init_transformations(chkp['transformations'])
        self._init_latents(chkp['latents'])
        if self.rank == 0:
            print(f'Loaded state from {chkp_path}')
    
    def _init_atlas_training(self):
        self.datasets, self.dataloaders = {}, {}
        self.inr_decoder, self.latents, self.transformations = {}, {}, {}
        self.optimizers, self.grad_scalers = {}, {}
        self.schedulers = {}
        chkp_path = self.args['load_model']['path']
        if len(chkp_path) > 0:
            self.load_checkpoint(chkp_path, self.args['load_model']['epoch'])
        else:
            self._init_dataloading(split='train')
            self._init_inr(split='train')
            self._init_transformations(split='train')
            self._init_latents(split='train')
        self._init_optimizer(split='train') 
        self._init_dataloading(split='val')

    def _init_validation(self):
        self._seed()
        self._init_latents(split='val')
        self._init_transformations(split='val')
        self._init_optimizer(split='val')
        
        # 复制模型权重 (Unwrap)
        model_train = self.inr_decoder['train']
        if isinstance(model_train, DDP): model_train = model_train.module
            
        self.inr_decoder['val'] = copy.deepcopy(model_train)
        self.inr_decoder['val'].eval()

    def _init_transformations(self, tfs=None, split='train'):
        shape = (len(self.datasets[split]), max(self.args['inr_decoder']['tf_dim'], 6)) 
        tfs = torch.zeros(shape).to(self.device) if tfs is None else tfs.to(self.device)
        self.transformations[split] = nn.Parameter(tfs) if self.args['inr_decoder']['tf_dim'] > 0 else tfs 
        
    def _init_latents(self, lats=None, split='train'):
        shape = (len(self.datasets[split]), *self.args['inr_decoder']['latent_dim'])
        lats = torch.normal(0, 0.01, size=shape).to(self.device) if lats is None else lats.to(self.device)
        self.latents[split] = nn.Parameter(lats)
        if split == 'val': 
            shape_cond_val = (len(self.datasets['val']), self.args['inr_decoder']['cond_dims'])
            self.conditions_val = nn.Parameter(torch.normal(0, 0.01, size=shape_cond_val).to(self.device))

    def re_init_latents(self, split='train'):
        self.latents[split].data.normal_(0, 0.01)
        self.transformations[split].data.zero_()
        self.optimizers[split].zero_grad()
        # [注意] 如果这里 re-init，必须立刻在此处调用 synchronize_latents
        # 或者 broadcast，否则各 GPU 初始值不同步
        
    def _init_optimizer(self, split='train'):
        params = [{'name': f'latents_{split}',
                   'params': self.latents[split],
                   'lr': self.args['optimizer']['lr_latent'],
                   'weight_decay': self.args['optimizer']['latent_weight_decay']}]
        
        if self.args['inr_decoder']['tf_dim'] > 0:
            params.append({'name': f'transformations_{split}',
                           'params': self.transformations[split],
                           'lr': self.args['optimizer']['lr_tf'],
                           'weight_decay': self.args['optimizer']['tf_weight_decay']})
        if split == 'train':
            params.append({'name': f'inr_decoder',
                           'params': self.inr_decoder[split].parameters(),
                           'lr': self.args['optimizer']['lr_inr'],
                           'weight_decay': self.args['optimizer']['inr_weight_decay']})
        if split == 'val':
            params.append({'name': f'conditions_val',
                           'params': self.conditions_val,
                           'lr': self.args['optimizer']['lr_latent'],
                           'weight_decay': self.args['optimizer']['latent_weight_decay']})
        self.optimizers[split] = optim.AdamW(params)
        self.grad_scalers[split] = GradScaler() if self.args['amp'] else None
        if self.args['optimizer']['scheduler']['type'] == 'cosine':
            self.schedulers[split] = CosineAnnealingLR(self.optimizers[split], T_max=self.args['epochs'][split], 
                                                       eta_min=self.args['optimizer']['scheduler']['eta_min'])
        else:
            self.schedulers[split] = None

    def _update_scheduler(self, split='train'):
        if self.schedulers[split] is not None:
            self.schedulers[split].step()

    def _seed(self):
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        np.random.seed(self.args['seed'])
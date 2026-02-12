
import wandb as wd
import pandas as pd
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch.amp.grad_scaler import GradScaler
from models.inr_decoder import INR_Decoder, LatentRegressor
from data_loading.dataset import Data
from utils import *


class AtlasBuilder:
    """
    Class to build an atlas from a training dataset.
    """
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        self.loss_criterion = Criterion(args).to(args['device'])
        self._init_atlas_training()
        self.train_on_data()

    def train_on_data(self):
        # 如果加载了预训练模型，先验证一次（这一行保持不变）
        if len(self.args['load_model']['path']) > 0: 
            self.validate(epoch_train=0) 
            
        loss_hist_epochs = []
        start_time = time.time()
        
        # 获取总 Epoch 数
        total_epochs = self.args['epochs']['train']
        
        for epoch in range(total_epochs):
            if self.args['optimizer']['re_init_latents']: 
                self.re_init_latents()
                
            loss = self.train_epoch(epoch, split='train')
            loss_hist_epochs.append(loss)
            
            print(f"Training: Epoch: {epoch}, Loss: {np.mean(loss_hist_epochs):.4f}, Total Time Epoch: {time.time() - start_time:.2f}s")

            if epoch > 0 and (epoch % self.args['validate_every'] == 0 or epoch == total_epochs - 1):
                self.validate(epoch)
                
            self._update_scheduler(split='train')
            
        return np.mean(loss_hist_epochs)

    def train_epoch(self, epoch, split):
        self.inr_decoder[split].train() if split == 'train' else self.inr_decoder[split].eval()
        loss_hist_batches = []
        time_data_loader = time.time()
        for batch in self.dataloaders[split]:
            print(f"Split: {split}, Current Epoch: {epoch}, Time Loading Batch: {time.time() - time_data_loader:.2f}s")
            start_time = time.time()
            loss = self.train_batch(batch, epoch, split)
            loss_hist_batches.append(loss)
            print(f"Split: {split}, Current Epoch: {epoch}, Loss Batch: {loss:.4f}, Total Training Time Batch: {time.time() - start_time:.2f}s")
        return np.mean(loss_hist_batches)
        
    def train_batch(self, batch, epoch, split='train'):
        loss_hist_samples = []
        n_smpls = self.args['n_samples']
        seg_weight = self.args['optimizer']['seg_weight'] if split == 'train' else 0.0
        coords_batch, values_batch, conditions_batch, idx_df_batch = to_device(batch)
        sample_iterator = range(0, idx_df_batch.shape[0], n_smpls)
        start_time = time.time()
        print(f"Split: {split}, Current Epoch: {epoch}, Starting Batch ...\n")
        
        for i, smpls in enumerate(sample_iterator):
            self.optimizers[split].zero_grad()
            coords = coords_batch[smpls:smpls + n_smpls]
            values = values_batch[smpls:smpls + n_smpls]
            idx_df = idx_df_batch[smpls:smpls + n_smpls].squeeze()
            # during validation we let the model predict the conditions
            conditions = conditions_batch[smpls:smpls + n_smpls] if split == 'train' else self.conditions_val[idx_df]

            with torch.autocast(device_type=self.device, enabled=self.args['amp']):
                # [修改 1] 解包返回值，获取预测值 values_p 和辅助损失 aux_loss (MoE load balancing loss)
                values_p, aux_loss = self.inr_decoder[split](coords, self.latents[split], conditions,
                                            self.transformations[split][idx_df], idcs_df=idx_df)
                
                # [修改 2] 将 aux_loss 传递给损失函数
                loss = self.loss_criterion(values_p, values, self.transformations[split][idx_df], 
                                           moe_loss=aux_loss,
                                           seg_weight=seg_weight)

            if self.args['amp']:    
                self.grad_scalers[split].scale(loss['total']).backward()
                
                # [AMP 模式下的梯度裁剪] (保留原有注释逻辑)
                #if split == 'train':
                    #self.grad_scalers[split].unscale_(self.optimizers[split])
                    #torch.nn.utils.clip_grad_norm_(self.inr_decoder[split].parameters(), max_norm=1.0)
                
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                loss['total'].backward()
                
                # [普通模式下的梯度裁剪] (保留原有注释逻辑)
                #if split == 'train':
                    #torch.nn.utils.clip_grad_norm_(self.inr_decoder[split].parameters(), max_norm=1.0)
                
                self.optimizers[split].step()

            loss_hist_samples.append(loss['total'].item())
            if i % 100 == 0 or i == (len(sample_iterator) - 1):
                log_loss(loss, epoch, split, self.args['logging'])
                print(f"Split: {split}, Epoch: {epoch}, "
                      f"Elapsed Training Time Batch: {time.time() - start_time:.2f}s"
                      f"Progress: {i/len(sample_iterator):.2f},"
                      f"Loss: {np.mean(loss_hist_samples):.4f},")
        return np.mean(loss_hist_samples)
        
    def validate(self, epoch_train):
        """
        Validate the model on the validation set.
        """
        # 1. 生成 Atlas (保持不变)
        if self.args['generate_cond_atlas']: 
            self.generate_atlas(epoch_train, n_max=100)

        # --------- Start Actual Validation ---------
        # [修改] 注释掉下面这行多余的判断 (或者改为 if True:)
        # if (epoch_train+1) % self.args['validate_every'] == 0 or ... :
        
        # 直接开始执行 inference
        print(f"Starting inference for Epoch {epoch_train}...") # 加个打印确认
        
        # Evaluate reconstruction quality on training subjects
        # 注意：这里可以保留对 train set 的采样逻辑
        num_train = len(self.datasets['train'])
        train_indices = [0, 2, 3] if num_train > 3 else list(range(num_train))
        metrics_train = self.generate_subjects_from_df(idcs_df=train_indices, epoch=epoch_train, split='train')
        log_metrics(self.args, metrics_train, epoch_train, df=self.datasets['train'].df, split='train')

        # Evaluate reconstruction quality on validation subjects
        self._init_validation() 
        
        # Validation Optimization Loop (Test-time Optimization)
        for epoch_val in range(self.args['epochs']['val']):
            self.train_epoch(epoch=epoch_val, split='val') 
            self._update_scheduler(split='val') 
            self.analyze_latent_space(epoch_train, epoch_val=epoch_val)
            
        metrics_val = self.generate_subjects_from_df(idcs_df=range(len(self.datasets['val'])), 
                                                    epoch=epoch_val, split='val')
        log_metrics(self.args, metrics_val, epoch_train, df=self.datasets['val'].df, split='val')
        
        self.save_state(epoch_train)

    def generate_subject_from_latent(self, latent_vec, condition_vector, transformation=None, split='train'):
        """
        Generates a subject from a latent vector, a condition vector and optional transformation parameters.
        """
        grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device, enabled=self.args['amp']):
                volume_inf = self.inr_decoder[split].inference(grid_coords, latent_vec, condition_vector, 
                                                        grid_shape, transformation)
        return volume_inf

    def generate_subjects_from_df(self, idcs_df=None, epoch=0, split='train'):
        """
        (Re)Generate subjects in their NATIVE space (aligned with original image).
        """
        import nibabel as nib 
        metrics = []
        
        # 定义一个辅助函数：生成原生网格
        def generate_native_grid(header_nii, world_bbox):
            # 1. 获取原图的尺寸和变换矩阵
            shape = header_nii.shape
            affine = header_nii.affine
            
            # 2. 生成原图每个体素的索引 (i, j, k)
            # 注意：这里生成的网格必须和原图 shape 一模一样
            i = torch.arange(0, shape[0], device=self.device)
            j = torch.arange(0, shape[1], device=self.device)
            k = torch.arange(0, shape[2], device=self.device)
            grid = torch.meshgrid(i, j, k, indexing='ij')
            grid_coords_idx = torch.stack(grid, dim=-1).reshape(-1, 3).float() # (N, 3)
            
            # 3. 将索引转换为物理坐标 (Native Physical Coordinates)
            # P = A * idx
            # 使用 torch 矩阵乘法加速
            affine_torch = torch.tensor(affine, dtype=torch.float32, device=self.device)
            # [N, 3] -> [N, 4] (homogenous)
            ones = torch.ones((grid_coords_idx.shape[0], 1), device=self.device)
            grid_coords_homo = torch.cat([grid_coords_idx, ones], dim=1)
            # P = (A @ idx.T).T
            grid_coords_phys = (affine_torch @ grid_coords_homo.T).T[:, :3]
            
            # 4. 执行和 Dataset.py 完全一致的归一化逻辑
            # training_coords = (phys - geometric_center) / (bbox / 2)
            
            # 计算几何中心 (Geometric Center)
            img_center_index = torch.tensor(shape, device=self.device) / 2.0
            # geometric_center = A * center_idx
            center_homo = torch.cat([img_center_index, torch.tensor([1.0], device=self.device)])
            geometric_center = (affine_torch @ center_homo)[:3]
            
            # 归一化
            # coords = phys - center
            grid_coords_norm = grid_coords_phys - geometric_center
            
            # Scale by bbox
            wb_torch = torch.tensor(world_bbox, dtype=torch.float32, device=self.device)
            grid_coords_norm = grid_coords_norm / (wb_torch / 2.0)
            
            return grid_coords_norm, list(shape), affine

        for idx_df in idcs_df:
            df_row_dict = self.datasets[split].df.iloc[idx_df].to_dict()
            
            ref_mod_path = df_row_dict[self.args['dataset']['modalities'][0]]
            ref_nii = nib.load(ref_mod_path)
            grid_coords, grid_shape, affine = generate_native_grid(
                ref_nii, 
                self.args['dataset']['world_bbox']
            )
            
            with torch.no_grad():
                transformations = self.transformations[split][idx_df, None]
                conditions = self.datasets[split].load_conditions(df_row_dict).to(self.device)
                
                # Inference
                volume_inf = self.inr_decoder[split].inference(
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
            """
            Generate temporal atlas for each condition combination.
            [Fixed] Removed .item() to fix AttributeError.
            """
            print(f"Generating atlases (depending on resolution and number of atlases this may take some time) ...\n")
            self.inr_decoder['train'].eval()
            grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
            temp_steps = self.args['atlas_gen']['temporal_values']
            atlas_list = []
            
            with torch.no_grad():
                for temp_step in temp_steps:
                    # 1. 计算归一化的年龄 (scan_age)
                    # normalize_condition 返回的是 float (如果输入是 float)
                    temp_step_normed = normalize_condition(self.args, 'scan_age', temp_step)
                    mean_latent = self.get_mean_latent('scan_age', temp_step_normed, n_max=n_max)
                    
                    # 2. 生成其他条件组合 (如 Sex)
                    condition_vectors = generate_combinations(self.args, self.args['atlas_gen']['conditions'])
                    
                    cond_list = []
                    for c_v in condition_vectors:
                        # [核心修复] 将 scan_age 拼接到条件向量中
                        # 如果 scan_age 是第一个条件
                        if self.args['dataset']['conditions'].get('scan_age', False):
                            # 修正点：直接使用 temp_step_normed，不要加 .item()
                            c_v = [temp_step_normed] + c_v
                        
                        # 转为 tensor
                        c_v = torch.tensor(c_v, dtype=torch.float32).to(self.device)
                        
                        # 推理
                        values_p = self.inr_decoder['train'].inference(grid_coords, mean_latent, c_v, 
                                                                    grid_shape, None)
                        seg = values_p[:, :, :, -1]
                        seg[seg==4] = 0
                        values_p[:, :, :, -1] = seg
                        cond_list.append(values_p.detach().cpu())
                        torch.cuda.empty_cache()
                        
                    atlas_list.append(torch.stack(cond_list, dim=-1))
                    
            atlas_list = torch.stack(atlas_list, dim=-1) 
            save_atlas(self.args, atlas_list, affine, temp_steps, condition_vectors, epoch=epoch)
            return atlas_list
    
    def get_mean_latent(self, condition_key, condition_mean, n_max=100, split='train'):
        """
        Regress gaussian weighted latent code from subjects weighted by distance to condition mean
        of the condition with condition_key. Weights are clipped to the closest n_max subjects.
        sigma is the standard deviation of the gaussian distribution used to weight the latents
        emperically we want +/- 2 stds (covering 95% of the weights) to span +/- "gaussian_span" weeks of scan age, e.g. 0.75 weeks.
        Therefore:
        - Full range of condition values is [-1, 1], i.e. 2. 
        - Full range of scan age is c_max - c_min = c_range, e.g. 46 - 37 = 9 for term neonates.
        - The ratio of condition values to weeks is 2 / c_range = c_ratio, e.g. 2 / 9 = 0.222 units per week.
        ==> 2 std = 0.75 weeks = 0.75 * c_ratio e.g. = 0.165 units.
        ==> sigma = 1 std = 0.5 * 0.75 weeks * c_ratio, e.g. = 0.0825 units for term neonates.
        # Finally, we scale the sigma by the condition scale factor in the config, as scan age is actually normalized to [-cond_scale, cond_scale]
        """
        c_ratio = 2 / (self.args['dataset']['constraints'][condition_key]['max'] - self.args['dataset']['constraints'][condition_key]['min'])
        span_weeks = self.args['atlas_gen']['gaussian_span']
        sigma = 0.5 * span_weeks * c_ratio
        sigma = sigma * self.args['atlas_gen']['cond_scale']

        latents = self.latents[split]
        condition_values, df_idcs = self.datasets[split].get_condition_values(condition_key, normed=True, device=self.device)
        assert len(condition_values) == len(latents), "Condition values (all entries from the dataframe) \
                                                       and latents must have the same length!"
        weights = torch.exp(-(condition_values - condition_mean)**2 / (2*(sigma**2)))
        n_max = min(n_max, len(weights))
        weights[torch.argsort(weights, descending=True)[n_max:]] = 0
        weights = weights / torch.sum(weights)
        weights = weights[:, None, None, None, None] # [n_subjects, *4D]
        mean_latent = torch.sum(latents * weights, dim=0, keepdim=True)
        return mean_latent
    
    def analyze_latent_space(self, epoch, epoch_val=0):
        args_LA = self.args['latent_anaylsis']
        args_C = self.args['dataset']['conditions']
        if not args_LA['activate']: return  # skip if not activated in config
        # conduct latent space analysis including
        # - birth age prediction from condition
        # - scan age prediction from latent_code
        if args_LA['predict_ext_condition'] != 'none':
            if args_C['birth_age'] and args_LA['predict_ext_condition'] == 'MAE':
                self.predict_ext_condition('birth_age', epoch, epoch_val)
            elif args_C['sex'] and args_LA['predict_ext_condition'] == 'CrossEntropy':
                self.predict_ext_condition('sex', epoch, epoch_val)
        
        if args_LA['predict_scan_age'] != 'none':
            self.predict_cond_value(epoch, epoch_val, cond_key='scan_age')
        
        # [Fix] 增加检查：只有当数据集配置中 birth_age 为 True 时才执行
        if args_LA['predict_birth_age'] != 'none' and args_C.get('birth_age', False):
            self.predict_cond_value(epoch, epoch_val, cond_key='birth_age')

        if args_LA['ba_regression']['activate'] and args_C.get('scan_age', False):
            self.regress_latent_condition('scan_age', epoch, epoch_val)
        # [Fix] 这里的回归分析也要加同样的检查
        # if args_LA['ba_regression']['activate'] and args_C.get('birth_age', False):
        #     self.regress_latent_condition('birth_age', epoch, epoch_val)

    def predict_cond_value(self, epoch, epoch_val=0, k=5, cond_key='scan_age'):
        # Predict scan_age using either NCA, PCA, SVR
        # 1. train regression network on training latents 
        # 2. after each epoch, compute regression on validation latents
        # 3. compute metrics
        # 4. log metrics
        train_data = self.latents['train'].detach().cpu().clone().numpy()
        train_data = train_data.reshape(train_data.shape[0], -1)
        train_labels = self.datasets['train'].get_condition_values(cond_key, normed=False, device=self.device)[0].cpu().numpy()
        train_labels_rnd = np.round(train_labels).astype(int)
        val_data = self.latents['val'].detach().cpu().clone().numpy()
        val_data = val_data.reshape(val_data.shape[0], -1)
        val_labels = self.datasets['val'].get_condition_values(cond_key, normed=False, device=self.device)[0].cpu().numpy()
        val_labels_rnd = np.round(val_labels).astype(int)

        nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=42))
        knn = KNeighborsClassifier(n_neighbors=min(min(len(self.latents['val']), len(self.latents['train'])), k))
        nca.fit(train_data, train_labels_rnd)
        knn.fit(nca.transform(train_data), train_labels_rnd)
        nca_predictions = knn.predict(nca.transform(val_data))
        nca_acc = np.mean(np.abs(nca_predictions - val_labels_rnd))
        nca_std = np.std(np.abs(nca_predictions - val_labels_rnd))

        print(f"Epoch_train {epoch}, Epoch_val {epoch_val}: Accuracy of NCA classifier for {cond_key}: {nca_acc:.3f} +/- {nca_std:.3f}")
        if self.args['logging']:
            wd.log({f'latent_anaylsis/{cond_key}_accuracy_nca': nca_acc})
        
    def predict_ext_condition(self, condition_key, epoch, epoch_val=0):
        # retrieve ground truth
        condition_values, df_idcs = self.datasets['val'].get_condition_values(condition_key, normed=False, device=self.device)
        cond_idx = list(self.args['atlas_gen']['conditions'].keys()).index(condition_key)
        condition_predictions = self.conditions_val[:, cond_idx]
        condition_predictions = denormalize_conditions(self.args, condition_key, condition_predictions)
        # compute metrics
        if self.args['latent_anaylsis']['predict_ext_condition'] == 'MAE':
            mae = torch.mean(torch.abs(condition_predictions - condition_values))
            value = mae.item()
            print(f"MAE for {condition_key}: {mae:.3f} at epoch {epoch}, epoch_val {epoch_val}")
        elif self.args['latent_anaylsis']['predict_ext_condition'] == 'CrossEntropy':
            ce = torch.nn.functional.cross_entropy(condition_predictions, condition_values)
            value = ce.item()
            print(f"Cross entropy for {condition_key}: {ce:.3f} at epoch {epoch}, epoch_val {epoch_val}")
        else:
            raise ValueError(f"Unknown metric {self.args['latent_anaylsis']['predict_ext_condition']}")
        if self.args['logging']:
            wd.log({f"latent_anaylsis/{condition_key}": value})

    def regress_latent_condition(self, condition_key, epoch_train=0, epoch_val=0):
            # 1. 获取并强制整形训练数据
            train_data = self.latents['train'].detach().clone()
            # [安全修复] 强制把 Latent 展平为 (N, 256)，防止被误认为是坐标
            train_data = train_data.view(train_data.shape[0], -1) 
            train_labels = self.datasets['train'].get_condition_values(condition_key, normed=True, device=self.device)[0]
            
            val_data = self.latents['val'].detach().clone()
            val_data = val_data.view(val_data.shape[0], -1)
            val_labels = self.datasets['val'].get_condition_values(condition_key, normed=True, device=self.device)[0]

            # 初始化回归器 (输入维度 256)
            latent_dim_size = train_data.shape[1]
            regressor = LatentRegressor([latent_dim_size]).to(self.device)
            
            optimizer = optim.AdamW(regressor.parameters(), lr=self.args['latent_anaylsis']['ba_regression']['lr'],
                                    weight_decay=0.0)
            batch_size = 32
            loss_fnc = torch.nn.L1Loss()
            regressor.train()
            
            regression_epochs = self.args['latent_anaylsis']['ba_regression']['epochs']
            best_score_val = float('inf')
            best_score_val_epoch = 0
            
            for epoch in range(regression_epochs):
                shuffle = np.random.permutation(len(train_data))
                train_data_sh = train_data[shuffle]
                train_labels_sh = train_labels[shuffle]
                loss_train_epoch = []
                
                for i in range(0, len(train_data_sh), batch_size):
                    train_data_batch = train_data_sh[i:i+batch_size]
                    train_labels_batch = train_labels_sh[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    
                    # [安全修复] 弃用危险的 .squeeze()，直接喂入正确的二维 Tensor (Batch, 256)
                    pred_train = regressor(train_data_batch).view(-1)
                    
                    # 确保 labels 也是一维的
                    loss_train = loss_fnc(pred_train, train_labels_batch.view(-1))
                    loss_train.backward()
                    optimizer.step()
                    loss_train_epoch.append(loss_train.item())
                    
                if epoch % 1 == 0:
                    print(f"Epoch {epoch} - Loss: {np.mean(loss_train_epoch):.4f}")
                    regressor.eval()
                    with torch.no_grad():   
                        pred_val = regressor(val_data).view(-1)
                        loss_val = loss_fnc(pred_val, val_labels.view(-1)) 
                        
                        # compute metrics
                        pred_val_denormed = denormalize_conditions(self.args, condition_key, pred_val)
                        val_labels_denormed = denormalize_conditions(self.args, condition_key, val_labels.view(-1))
                        mae = torch.mean(torch.abs(pred_val_denormed - val_labels_denormed))
                        
                        if mae < best_score_val:
                            best_score_val = mae
                            best_score_val_epoch = epoch
                            
                    if self.args['logging']:
                        wd.log({f"latent_anaylsis/{condition_key}_regression_train": loss_train.item()})
                        wd.log({f"latent_anaylsis/{condition_key}_regression_val": loss_val.item()})
                        wd.log({f"latent_anaylsis/{condition_key}_regression_mae_val": mae.item()})
                    regressor.train()
                    
            print(f"Best MAE for {condition_key}: {best_score_val:.3f} at regression_epoch {best_score_val_epoch} for epoch_train {epoch_train}, epoch_val {epoch_val}")

    def save_state(self, epoch, split='train'):
        if self.args['save_model']:
            log_dir = self.args['output_dir']
            torch.save({
                'epoch': epoch,
                'latents': self.latents[split].cpu(),
                'transformations': self.transformations[split].cpu(),
                'inr_decoder': self.inr_decoder[split].state_dict(),
                'tsv_file': self.datasets[split].tsv_file,
                'dataset_df': self.datasets[split].df,
                'args': self.args
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Saved model state to {os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")}')
        else:
            print(f'Not saving model state as save_model is set to False')

    def load_checkpoint(self, chkp_path=None, epoch=None):  
        chkp_path = os.path.join(chkp_path, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f'State file {chkp_path} not found!')
        chkp = torch.load(chkp_path, weights_only=False)
        # self.args = chkp['args']
        self._init_dataloading(chkp['tsv_file'], chkp['dataset_df'])
        self._init_inr(chkp['inr_decoder'], split='train')
        self._init_transformations(chkp['transformations'])
        self._init_latents(chkp['latents'])
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
        self._init_optimizer(split='train') # optimizer is not loaded from checkpoint
        self._init_dataloading(split='val')

    def _init_validation(self):
        self._seed()
        self._init_latents(split='val')
        self._init_transformations(split='val')
        self._init_optimizer(split='val')
        self.inr_decoder['val'] = copy.deepcopy(self.inr_decoder['train'])
        self.inr_decoder['val'].eval()

    def _init_dataloading(self, tsv_file=None, df_loaded=None, split='train'):
        shuffle = True if split == 'train' else False
        tsv_file =  pd.read_csv(self.args['dataset']['tsv_file'], sep='\t') if tsv_file is None else tsv_file
        self.datasets[split] = Data(self.args, tsv_file, split=split, df_loaded=df_loaded)
        self.dataloaders[split] = DataLoader(self.datasets[split], batch_size=self.args['batch_size'], 
                                             num_workers=self.args['num_workers'], shuffle=shuffle, 
                                             collate_fn=self.datasets[split].collate_fn, pin_memory=True)

        print(f"Initialized dataloader for {split} with {len(self.datasets[split])} subjects")

    def _init_inr(self, state_dict=None, split='train'):
        # get the number of active conditions
        self.args['inr_decoder']['cond_dims'] = sum([self.args['dataset']['conditions'][c] 
                                                     for c in self.args['dataset']['conditions']])
        self.inr_decoder[split] = INR_Decoder(self.args, self.device).to(self.device)
        if state_dict is not None:
            self.inr_decoder[split].load_state_dict(state_dict)

    def _init_transformations(self, tfs=None, split='train'):
        shape = (len(self.datasets[split]), max(self.args['inr_decoder']['tf_dim'], 6)) # at least 6 for rigid, 9 for rigid+scale
        tfs = torch.zeros(shape).to(self.device) if tfs is None else tfs.to(self.device)
        self.transformations[split] = nn.Parameter(tfs) if self.args['inr_decoder']['tf_dim'] > 0 else tfs # if tf_dim=0, set trafos to 0 and fix
        
    def _init_latents(self, lats=None, split='train'):
        shape = (len(self.datasets[split]), *self.args['inr_decoder']['latent_dim'])
        lats = torch.normal(0, 0.01, size=shape).to(self.device) if lats is None else lats.to(self.device)
        self.latents[split] = nn.Parameter(lats)
        if split == 'val': # need to initialize conditions as learnable parameters
            shape_cond_val = (len(self.datasets['val']), self.args['inr_decoder']['cond_dims'])
            self.conditions_val = nn.Parameter(torch.normal(0, 0.01, size=shape_cond_val).to(self.device))

    def re_init_latents(self, split='train'):
        self.latents[split].data.normal_(0, 0.01)
        self.transformations[split].data.zero_()
        self.optimizers[split].zero_grad()
        
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
    
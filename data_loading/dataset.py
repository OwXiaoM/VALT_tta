import os
import math
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import pandas as pd
import torchio as tio
from torch.utils.data import Dataset
from utils import *


class Data(Dataset):
    def __init__(self, args, tsv_file, split, df_loaded=None):
        self.args = args
        self.split = split
        self.modality_keys = self.args['dataset']['modalities']
        self.world_bbox = np.array(self.args['dataset']['world_bbox'])
        self.tsv_file = tsv_file
        self.df = self.filter_dataframe(self.tsv_file) if df_loaded is None else df_loaded
        
        # [修改 1] 开启内存缓存
        # 只要内存足够，这将显著加速第二轮 Epoch 之后的训练
        self.cache = {}
        self.use_cache = False 
        if self.split == 'train':
            print(f"[{split}] Data Caching Enabled: {self.use_cache}")

        self._init_data_augmentation()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # [修改 2] 缓存读取逻辑
        # 如果开启缓存且已存在，直接使用（避免重复 IO 和 预处理）
        if self.use_cache and idx in self.cache:
            modalities = self.cache[idx]
        else:
            row_dict = self.df.iloc[idx].to_dict()
            modalities = self.load_modalities(row_dict)
            if self.use_cache:
                self.cache[idx] = modalities # 存入缓存

        # 注意：load_coords_and_values 内部会调用 augment_modalities
        # augment 是随机的，所以每次即使输入同样的 modalities，输出的 coords/values 也会不同（符合预期）
        coords, values = self.load_coords_and_values(modalities)
        
        coords = torch.tensor(coords, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        
        # 获取 conditions (需要重新获取 row_dict 因为缓存里只存了 modalities)
        row_dict = self.df.iloc[idx].to_dict()
        conditions = self.load_conditions(row_dict)[None, :].expand(coords.shape[0], -1)
        idx_df = torch.tensor(idx, dtype=torch.int32).unsqueeze(0).expand(coords.shape[0], -1)
        return coords, values, conditions, idx_df

    def collate_fn(self, batch, shuffle=True):
        coords = torch.concat([b[0] for b in batch], dim=0)
        values = torch.concat([b[1] for b in batch], dim=0)
        conditions = torch.concat([b[2] for b in batch], dim=0)
        idx_df = torch.concat([b[3] for b in batch], dim=0)
        if shuffle:
            perm = torch.randperm(coords.shape[0])
            coords = coords[perm]
            values = values[perm]
            conditions = conditions[perm]
            idx_df = idx_df[perm]
        return coords, values, conditions, idx_df

    def load_modalities(self, row_dict):
        modalities = {}
        for i, mod_key in enumerate(self.modality_keys):
            if mod_key not in row_dict:
                raise ValueError(f"Modality {mod_key} not found in row_dict")
            if row_dict[mod_key] == "":
                raise ValueError(f"Modality {mod_key} is empty")
            modalities[mod_key] = nib.load(row_dict[mod_key])
            #modalities[mod_key] = squeeze_modalities(modalities[mod_key], path=row_dict[mod_key], write2disk=True)
            modalities[mod_key] = squeeze_modalities(modalities[mod_key], path=row_dict[mod_key], write2disk=False)
        # check that all modalities have the same shape
        shapes = [modalities[mod_key].shape for mod_key in modalities]
        if len(set(shapes)) != 1:
            raise ValueError(f"Modalities have different shapes: {shapes}")
        
        # check that all modalities have approximately the same affine
        affines = [modalities[mod_key].affine for mod_key in modalities]
        for i, affine in enumerate(affines):
            if i == 0 or np.allclose(affines[i], affines[i-1], atol=1e-2):
                continue
            else:
                raise ValueError(f"Modalities have different affines: {affines}")

        # if mask_reconstruction is enabled, mask the modalities and add background halo 
        # to the segmentation. This helps to delineate the brain from the (noisy) background.
        if self.args['mask_reconstruction']:
            mask = modalities[self.modality_keys[-1]].get_fdata()>0
            for mod_key in self.modality_keys[:-1]:
                modalities[mod_key] = mask_nifti(modalities[mod_key], mask)
            modalities[self.modality_keys[-1]] = add_background_halo(self.args['dataset']['label_names'], 
                                                                      modalities[self.modality_keys[-1]])
        
        return modalities

    def load_coords_and_values(self, modalities, normalize=True):
        # 1. 数据增强与加载 (保持不变)
        modalities_data = self.augment_modalities(modalities)
        mra_data = modalities_data[self.modality_keys[0]]
        seg_data = modalities_data[self.modality_keys[-1]]
        affine = modalities[self.modality_keys[-1]].affine
        img_shape = mra_data.shape

        # --------------------------------------------------------------
        # 2. 前景采样 (Vessel) - 保持全采
        # --------------------------------------------------------------
        vessel_indices = np.argwhere(seg_data > 0)
        n_vessel = len(vessel_indices)
        
        # [修改点 1] 设定一个固定的、巨大的背景目标
        # 不管血管有多少，我们都强制要求采够 500万个背景点
        # 这样能保证每个 Batch 的数据量足够让 GPU 训练很久
        target_n_bg = 5000000 

        # --------------------------------------------------------------
        # 3. 背景采样 (极速拒绝采样)
        # --------------------------------------------------------------
        if n_vessel > 0:
            fg_mean = np.mean(mra_data[seg_data > 0])
            air_threshold = fg_mean * 0.1 
        else:
            air_threshold = 0.01

        selected_bg_indices = []
        

        n_candidates = target_n_bg * 2
        rand_coords = np.random.randint(0, [img_shape[0], img_shape[1], img_shape[2]], size=(n_candidates, 3))
        
        # 批量获取像素值 (Numpy 高级索引)
        vals_mra = mra_data[rand_coords[:,0], rand_coords[:,1], rand_coords[:,2]]
        vals_seg = seg_data[rand_coords[:,0], rand_coords[:,1], rand_coords[:,2]]
        
        # [关键] 这里保证了不是“乱采”
        # 我们依然严格区分 Tissue 和 Air
        is_bg = (vals_seg == 0)
        mask_tissue = is_bg & (vals_mra > air_threshold)
        mask_air = is_bg & (vals_mra <= air_threshold)
        
        pool_tissue = rand_coords[mask_tissue]
        pool_air = rand_coords[mask_air]

        # 执行分层配额 (Tissue/Air 各一半)
        if len(pool_tissue) > 0 and len(pool_air) > 0:
            n_tissue_target = target_n_bg // 2
            n_air_target = target_n_bg - n_tissue_target
            
            # 够就截断，不够就全拿
            if len(pool_tissue) >= n_tissue_target:
                selected_bg_indices.append(pool_tissue[:n_tissue_target])
            else:
                selected_bg_indices.append(pool_tissue)
                
            if len(pool_air) >= n_air_target:
                selected_bg_indices.append(pool_air[:n_air_target])
            else:
                selected_bg_indices.append(pool_air)
                
        elif len(pool_tissue) > 0:
            selected_bg_indices.append(pool_tissue[:target_n_bg])
        elif len(pool_air) > 0:
            selected_bg_indices.append(pool_air[:target_n_bg])
            
        # 保底逻辑 (万一随机撒点没撒够，回退到慢速方法补齐，极少发生)
        if len(selected_bg_indices) == 0:
            fallback = np.argwhere(seg_data == 0)
            if len(fallback) > 0:
                indices = np.random.choice(len(fallback), min(len(fallback), target_n_bg))
                selected_bg_indices.append(fallback[indices])
        else:
            selected_bg_indices = np.vstack(selected_bg_indices)

        # --------------------------------------------------------------
        # 4. 合并前景和背景
        # --------------------------------------------------------------
        if len(selected_bg_indices) > 0:
            c_nz = np.vstack((vessel_indices, selected_bg_indices))
        else:
            c_nz = vessel_indices

        # --------------------------------------------------------------
        # 5. [修改点 2] 提升截断上限
        # --------------------------------------------------------------
        # 既然我们辛苦采了 500万背景，就允许它们都通过
        max_points_per_subject = 5000000 
        
        if len(c_nz) > max_points_per_subject:
            perm = np.random.permutation(len(c_nz))
            c_nz = c_nz[perm[:max_points_per_subject]]

        # --------------------------------------------------------------
        # 6. 提取值与坐标计算 (保持不变)
        # --------------------------------------------------------------
        values = np.stack([modalities_data[mod][c_nz[:, 0], c_nz[:, 1], c_nz[:, 2]].flatten() 
                            for mod in self.modality_keys], axis=-1)
        
        c_nz_phys = nib.affines.apply_affine(affine, c_nz)
        img_center_index = np.array(img_shape) / 2.0
        geometric_center = nib.affines.apply_affine(affine, img_center_index)
        
        coords = c_nz_phys - geometric_center
        
        if normalize:
            wb_center = self.world_bbox / 2
            coords = (coords / wb_center) 
            values = normalize_intensities(values, self.args['dataset']['normalize_values'])
            
        return coords, values

    # def load_coords_and_values(self, modalities, normalize=True):
    #     import scipy.ndimage as ndi # 确保引入了这个库

    #     # 1. 加载数据
    #     modalities_data = self.augment_modalities(modalities)
    #     mra_data = modalities_data[self.modality_keys[0]]
    #     seg_data = modalities_data[self.modality_keys[-1]]
    #     affine = modalities[self.modality_keys[-1]].affine
    #     img_shape = mra_data.shape

    #     # 强制清洗背景 (保险起见)
    #     mra_data[seg_data == 0] = 0.0

    #     # ==============================================================
    #     # [Step 1] 核心：获取血管点 (Positive Samples)
    #     # ==============================================================
    #     vessel_mask = (seg_data > 0)
    #     vessel_indices = np.argwhere(vessel_mask)
    #     n_vessel = len(vessel_indices)

    #     # ==============================================================
    #     # [Step 2] 关键改进：生成边界区域 (Hard Negative Mining)
    #     # ==============================================================
    #     # 逻辑：对血管 Mask 进行膨胀，然后减去原本的血管 Mask
    #     # 剩下的就是紧贴着血管的那一圈“壳” (Shell)
        
    #     # iterations=2 表示向外扩张约 2 个像素，你可以根据需要调整(1-3均可)
    #     dilated_mask = ndi.binary_dilation(vessel_mask, iterations=2)
        
    #     # 边界 = 膨胀后 有值 且 原本 无值 的地方
    #     boundary_mask = dilated_mask & (~vessel_mask)
    #     boundary_indices = np.argwhere(boundary_mask)
    #     n_boundary = len(boundary_indices)

    #     # ==============================================================
    #     # [Step 3] 制定采样配额 (4:4:2 策略)
    #     # ==============================================================
    #     # 我们希望重点攻克边界，所以让边界点的数量接近血管点的数量
        
    #     target_n_boundary = int(n_vessel * 1.0) # 1:1 强攻边界
    #     target_n_global   = int(n_vessel * 0.5) # 0.5 远场防幻觉
        
    #     # --- A. 采样边界点 ---
    #     if n_boundary > 0:
    #         if n_boundary >= target_n_boundary:
    #             # 边界点通常很多，随机选一部分
    #             idx = np.random.choice(n_boundary, target_n_boundary, replace=False)
    #             selected_boundary = boundary_indices[idx]
    #         else:
    #             # 如果血管极细，边界点不够，就全都要
    #             selected_boundary = boundary_indices
    #     else:
    #         selected_boundary = np.empty((0, 3), dtype=int)

    #     # --- B. 采样远场背景 (Global) ---
    #     # 随机撒点，只要不是血管也不是边界
    #     n_candidates = int(target_n_global * 3) + 10000
    #     rand_coords = np.random.randint(0, [img_shape[0], img_shape[1], img_shape[2]], 
    #                                     size=(n_candidates, 3))
        
    #     # 筛选：既不是血管(dilated_mask包含了血管和边界)的点
    #     # 也就是在 dilated_mask 之外的点
    #     vals_dilated = dilated_mask[rand_coords[:,0], rand_coords[:,1], rand_coords[:,2]]
    #     is_global_bg = (vals_dilated == 0)
    #     selected_global = rand_coords[is_global_bg]

    #     if len(selected_global) > target_n_global:
    #         selected_global = selected_global[:target_n_global]

    #     # ==============================================================
    #     # [Step 4] 三路合并
    #     # ==============================================================
    #     parts = [vessel_indices]
    #     if len(selected_boundary) > 0:
    #         parts.append(selected_boundary)
    #     if len(selected_global) > 0:
    #         parts.append(selected_global)
            
    #     c_nz = np.vstack(parts)
            
    #     # ==============================================================
    #     # [Step 5] 内存复制 (加速 IO) - 保持不变
    #     # ==============================================================
    #     target_total_points = 5000000 
    #     current_points = len(c_nz)
        
    #     if current_points < target_total_points and current_points > 0:
    #         repeat_factor = target_total_points // current_points + 1
    #         c_nz = np.tile(c_nz, (repeat_factor, 1))
    #         c_nz = c_nz[:target_total_points]
        
    #     np.random.shuffle(c_nz)

    #     # ==============================================================
    #     # [Step 6] 提取值 (保持不变)
    #     # ==============================================================
    #     values = np.stack([modalities_data[mod][c_nz[:, 0], c_nz[:, 1], c_nz[:, 2]].flatten() 
    #                         for mod in self.modality_keys], axis=-1)
        
    #     c_nz_phys = nib.affines.apply_affine(affine, c_nz)
    #     img_center_index = np.array(img_shape) / 2.0
    #     geometric_center = nib.affines.apply_affine(affine, img_center_index)
        
    #     coords = c_nz_phys - geometric_center
        
    #     if normalize:
    #         wb_center = self.world_bbox / 2
    #         coords = (coords / wb_center) 
    #         values = normalize_intensities(values, self.args['dataset']['normalize_values'])
            
    #     return coords, values

    def augment_modalities(self, modalities):
        if self.data_augmentation:
            tio_sub = {}
            for mod_key in modalities:
                data = modalities[mod_key].get_fdata()
                affine = modalities[mod_key].affine
                tensor_data = torch.from_numpy(data).type(torch.float32).unsqueeze(0)
                if "Seg" in mod_key:
                    tio_sub[mod_key] = tio.LabelMap(tensor=tensor_data, affine=affine)
                else:
                    tio_sub[mod_key] = tio.ScalarImage(tensor=tensor_data, affine=affine)
            
            tio_sub = tio.Subject(tio_sub)
            # modalities_data = {mod_key: tio_sub[mod_key].data.squeeze().numpy() for mod_key in modalities}
            # for mod_key in modalities_data:
            #     path_out_before = os.path.join(self.args['output_dir'], f"before_augmentation_{self.split}_{mod_key}.nii.gz")
            #     nib.save(nib.Nifti1Image(modalities_data[mod_key], affine), path_out_before)
            tio_sub = self.tio_transform(tio_sub)
            modalities_data = {mod_key: tio_sub[mod_key].data.squeeze().numpy() for mod_key in modalities}
            # for mod_key in modalities_data:
            #     path_out_after = os.path.join(self.args['output_dir'], f"after_augmentation_{self.split}_{mod_key}.nii.gz")
            #     nib.save(nib.Nifti1Image(modalities_data[mod_key], affine), path_out_after)
        else:
            modalities_data = {mod_key: modalities[mod_key].get_fdata() for mod_key in modalities}
        return modalities_data

    def load_conditions(self, row_dict, normalize=True):
        conditions = []
        for key in self.args['dataset']['conditions']:
            if self.args['dataset']['conditions'][key]: # if condition is enabled
                constraint = self.args['dataset']['constraints'][key]
                val = row_dict[key]

                # [修改] 自动处理分类变量 (Categoric)
                if constraint.get('type') == 'categoric':
                    # 获取定义的值列表，例如 ['M', 'F']
                    possible_values = constraint['values']
                    # 将字符串映射为索引: 'M'->0.0, 'F'->1.0
                    if val in possible_values:
                        val = float(possible_values.index(val))
                    else:
                        # 如果遇到未定义的值（如Unknown），默认为0
                        val = 0.0
                    
                    # 设定分类变量的 min/max 为索引范围
                    c_min = 0.0
                    c_max = float(len(possible_values) - 1)
                
                # [修改] 处理数值变量 (Numeric)
                else:
                    # 确保是浮点数
                    val = float(val)
                    # 从配置读取 min/max
                    c_min = constraint['min']
                    c_max = constraint['max']

                if normalize:
                    # 执行归一化公式: 映射到 [-scale, +scale]
                    if c_max > c_min:
                        norm_val = (((val - c_min) / (c_max - c_min)) * 2 - 1) * self.args['atlas_gen']['cond_scale']
                    else:
                        norm_val = 0.0 # 防止除以零
                    
                    row_dict[key] = norm_val
                else:
                    row_dict[key] = val
                
                conditions.append(row_dict[key])
        
        return torch.tensor(conditions, dtype=torch.float32)
            
    def filter_dataframe(self, df):
        '''
        Filter the dataframe based on the constraints in the args.
        Also check for missing modalities and remove subjects with missing modalities.
        Returns a dataframe with the filtered data, i.e. the final subjects
        that will be used for training, validation, or testing. 
        '''
        print("--------------------------------")
        print("Sampling data for split ", self.split, "\n")
        df = self.sample_subject_ids(df)
        df = self.remove_missing_modalities(df)
        df = self.check_constraints(df)
        df = self.sample_subjects(df)
        print("Sampled ", len(df), " subjects for split ", self.split, "\n")
        print("--------------------------------")
        return df

    def sample_subject_ids(self, df, verbose=True):
        """
        Sample subjects from subject_ids list provided by the args_data['subject_ids']
        """
        if self.args['dataset']['subject_ids'][self.split]:
            print(f"Sampling subjects from subject_ids list. Number of subjects in dataframe: {len(df)}, "
                  f"Number of subjects in subject_ids list: {len(self.args['dataset']['subject_ids'][self.split])}")
            df = df[df['subject_id'].isin(self.args['dataset']['subject_ids'][self.split])]
            print(f"Number of subjects in dataframe after sampling: {len(df)} \n")
        return df

    def remove_missing_modalities(self, df, verbose=True):
        """
        Removes rows if any required modality entry is missing or empty in the DataFrame.
        """
        if verbose:
            print("Initial number of subjects:", len(df))

        modalities = self.args['dataset']['modalities']
        # Start by assuming we keep every row
        keep_mask = np.ones(len(df), dtype=bool)

        for modality in modalities:
            if modality not in df.columns:
                print(f"Warning: modality column '{modality}' not in DataFrame. Skipping.")
                continue

            # Check that each entry is not null/NaN and is not an empty string
            col_values = df[modality]
            is_valid = col_values.notnull() & (col_values != "")
            keep_mask &= is_valid.to_numpy(dtype=bool)

        dropped_count = (~keep_mask).sum()
        if dropped_count > 0 and verbose:
            print(f"Dropping {dropped_count} subjects missing modality entries.")

        df = df[keep_mask].reset_index(drop=True)
        if verbose:
            print("Number of subjects after removing missing modalities:", len(df))

        return df

    def check_constraints(self, df, verbose=True):
        """
        Drops rows if any numeric constraint is outside [min, max].
        """
        if verbose:
            print("Initial number of subjects:", len(df))

        constraints_dict = self.args['dataset'].get('constraints', {})

        # Build a mask to keep all that pass constraints
        keep_mask = np.ones(len(df), dtype=bool)

        for ckey, cinfo in constraints_dict.items():
            if cinfo.get('type') == 'numeric':
                cmin = cinfo.get('min', None)
                cmax = cinfo.get('max', None)
                if cmin is not None and cmax is not None and ckey in df.columns:
                    vals = df[ckey].values
                    this_mask = (vals >= cmin) & (vals <= cmax)
                    keep_mask &= this_mask
            elif cinfo.get('type') == 'categoric':
                if ckey in df.columns:
                    this_mask = df[ckey].isin(cinfo.get('values'))
                    keep_mask &= this_mask
                else:
                    raise ValueError(f"Constraint column {ckey} not found in dataframe")
            else:
                raise ValueError(f"Constraint type {cinfo.get('type')} not supported")

        dropped_count = np.count_nonzero(~keep_mask)
        if dropped_count > 0 and verbose:
            print(f"Dropping {dropped_count} subjects outside constraint ranges.")
        df = df[keep_mask].reset_index(drop=True)

        if verbose:
            print("Number of subjects after constraints check:", len(df))

        return df
    
    def get_condition_values(self, condition_key, normed=True, device=None):
        """
        Get all values together with their entry_idx of a condition_key from the dataframe.
        """
        values = torch.tensor(self.df[condition_key].values, dtype=torch.float32, device=device)
        if normed:
            values = normalize_condition(self.args, condition_key, values)
        return values, torch.arange(len(self.df), device=device)

    def sample_subjects(self, df, verbose=True):
            """
            Nested sampling across constraints that have a 'distribution' with 'priority'.
            """
            if self.args['n_subjects'][self.split] > len(df):
                print("max_num_subjects must be greater than the number of subjects in the dataframe")
                print(f"max_num_subjects: {self.args['n_subjects'][self.split]}, number of subjects in the dataframe: {len(df)}")
                self.args['n_subjects'][self.split] = len(df)

            max_num_subjects = self.args['n_subjects'][self.split]
            # Gather constraints that define a distribution + priority
            constraints_with_prio = []
            for cname, cinfo in self.args['dataset'].get('constraints', {}).items():
                dist_info = cinfo.get('distribution', {})
                if 'priority' in dist_info:  # we only consider constraints that define a priority
                    constraints_with_prio.append((cname, cinfo))

            # Sort by ascending priority (higher = first).
            constraints_with_prio.sort(key=lambda x: x[1]['distribution']['priority'], reverse=True)

            if verbose:
                print(f"[sample_subjects] # subjects before sampling: {len(df)}")
                print("[sample_subjects] Constraints in priority order:",
                    [c[0] for c in constraints_with_prio])

            # Convert to list for recursion 
            constraints_list = [(cname, cinfo) for (cname, cinfo) in constraints_with_prio]
            # Sampling currently only uniformly over the highest priority constraint, random over the rest
            df_sampled = self._shallow_sampling(df, constraints_list, max_num_subjects, verbose)

            # Global cap if still too large
            if len(df_sampled) > max_num_subjects:
                df_sampled = df_sampled.sample(n=max_num_subjects, random_state=42).reset_index(drop=True)
                if verbose:
                    print(f"[sample_subjects] Truncated final set to max_num_subjects={max_num_subjects}")

            if verbose:
                print(f"[sample_subjects] # subjects after nested sampling: {len(df_sampled)}")

            # Print & Save histogram if verbose
            if verbose:
                for constraint in constraints_list:
                    self._print_and_save_hist(df_sampled, constraint)

            return df_sampled

    def _shallow_sampling(self, df, constraints_list, max_num_subjects, verbose=True):
        """
        Sample uniformly over the highest priority constraint. No nested sampling supported just yet.
        """
        current_constraint = constraints_list[0]
        current_constraint_name = current_constraint[0]
        current_constraint_info = current_constraint[1]
        c_min = current_constraint_info.get('min', None)
        c_max = current_constraint_info.get('max', None)
        bins = current_constraint_info.get('distribution').get('bins')
        if bins is None:
            bins = (c_max - c_min)
        # create bins if not specified
        edges = np.linspace(c_min, c_max, bins+1)
        edges[-1] += 1e-6  # ensures inclusive of c_max
        values = df[current_constraint_name].values
        bin_idx = np.digitize(values, edges) - 1  # bin indices in 0..(bins-1)
        bin_sizes = np.bincount(bin_idx)
        bin_labels = [f"[{edges[i]},{edges[i+1]}]" for i in range(len(edges)-1)]
        if verbose:
            print(f"Bin labels: {bin_labels}")
            print(f"Bin sizes: {bin_sizes}")
        
        # sample from each bin
        samples_drawn_per_bin = [0] * bins
        drained_bins = []
        while np.sum(samples_drawn_per_bin) < max_num_subjects: # sample until max_num_subjects is reached
            num_remaining_bins = bins - len(np.unique(drained_bins)) # number of bins left to sample from
            num_remaining_subjects = max_num_subjects - np.sum(samples_drawn_per_bin)
            required_samples_per_bin = math.ceil(num_remaining_subjects / num_remaining_bins) # calculate how many subjects to sample from each remaining bin to reach max_num_subjects
            print(f"Required (additional) samples per remaining bin: {required_samples_per_bin}")
            # sample from each bin the required number of subjects
            for i in range(bins):
                bin_df = df[bin_idx == i]
                if i in drained_bins: # if the bin is already drained, skip it
                    continue
                n_samples = min(required_samples_per_bin, len(bin_df)-samples_drawn_per_bin[i]) # sample from the bin the required number of subjects, if the bin is too small, sample all remaining subjects
                samples_drawn_per_bin[i] += n_samples
                if samples_drawn_per_bin[i] == len(bin_df): # if the bin is now drained, add it to the list of drained bins
                    drained_bins.append(i)
        print(f"Samples drawn per bin: {samples_drawn_per_bin}")
        df_sampled = []
        for i, n_samples in enumerate(samples_drawn_per_bin):
            if n_samples > 0:
                assert len(df[bin_idx == i]) >= n_samples
                df_sampled.append(df[bin_idx == i].sample(n=n_samples, random_state=42))
        df_sampled = pd.concat(df_sampled, ignore_index=True)
        return df_sampled


    def _print_and_save_hist(self, df, constraint):
        """
        Print the histogram data (counts per bin) for `column_name` in `df`,
        and save a bar plot as an image. Uses 30 bins by default, or fewer if df is small.
        """
        indent = "  "
        current_cname, current_cinfo = constraint
        # Drop NaNs if they exist
        col_data = df[current_cname].dropna().values
        if len(col_data) == 0:
            print(f"{indent}[Histogram] No data for column '{current_cname}'")
            return

        dist_info = current_cinfo.get('distribution', {})
        dist_type = dist_info.get('type')
        c_min = current_cinfo.get('min', None)
        c_max = current_cinfo.get('max', None)
        bins = dist_info.get('bins')
        if bins is None:
            bins = (c_max - c_min + 1) if (c_min is not None and c_max is not None) else 10
        edges = np.linspace(c_min, c_max, bins)
        edges[-1] += 1e-9  # ensures inclusive of c_max

        counts, bin_edges = np.histogram(col_data, bins=edges)
        print(f"{indent}[Histogram] {current_cname} counts: {counts.tolist()}")
        print(f"{indent}[Histogram] {current_cname} bin_edges: {bin_edges.tolist()}")

        # Save a bar plot
        plt.figure()
        # Midpoints for bar chart
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # plot ticks for every bin center, with slight rotation of the labels
        plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9)
        plt.xticks(bin_centers, rotation=45)
        # plot every y-tick 
        plt.yticks(range(0, max(counts)+1))
        plt.title(f"Histogram of {current_cname} with {dist_type} distribution")
        plt.xlabel(current_cname)
        plt.ylabel("Count")

        out_path = os.path.join(self.args['output_dir'], f"hist_{current_cname}_dist_{dist_type}_{self.split}.png")
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()  # free memory
        print(f"{indent}Saved histogram to {out_path}")

    def _init_data_augmentation(self):
        """
        Initialize data augmentation.
        """
        self.data_augmentation = self.args['data_augmentation']['activate'] * (self.split == 'train')
        if self.data_augmentation:
            self.transforms = []
            args_aug = self.args['data_augmentation']
            if args_aug['augment_deformation']['p'] > 0.0:
                args = args_aug['augment_deformation']
                self.transforms.append(tio.RandomElasticDeformation(p=args['p'], num_control_points=args['num_control_points'],
                                                                    max_displacement=args['max_displacement']))
            if args_aug['augment_motion']['p'] > 0.0:
                args = args_aug['augment_motion']
                self.transforms.append(tio.RandomMotion(p=args['p'], degrees=args['degrees'],
                                                        translation=args['translation'],
                                                        num_transforms=args['num_transforms']))
            if args_aug['augment_noise']['p'] > 0.0:
                args = args_aug['augment_noise']
                self.transforms.append(tio.RandomNoise(p=args['p'], mean=args['mean'], std=args['std']))
            if args_aug['augment_biasfield']['p'] > 0.0:
                args = args_aug['augment_biasfield']
                self.transforms.append(tio.RandomBiasField(p=args['p'], coefficients=args['coeff']))
            if args_aug['augment_gamma']['p'] > 0.0:
                args = args_aug['augment_gamma']
                self.transforms.append(tio.RandomGamma(p=args['p'], log_gamma=args['log_gamma']))
            # Add the subject-level masking transform.
            # This transform uses the segmentation (e.g. "SegEM9", last modality) to mask intensity images. 
            self.transforms.append(MaskSubjectTransform(segmentation_key=self.modality_keys[-1]))
            self.tio_transform = tio.Compose(self.transforms) if self.transforms else None
        
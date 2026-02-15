import os
import glob
import yaml
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

# 引入项目模块
from models.inr_decoder import INR_Decoder
from data_loading.dataset import Data # 需要用到 Dataset 的一些辅助函数
from utils import generate_world_grid, normalize_condition

# ================= 0. 临床数据加载 =================
def load_clinical_data(excel_path=None):
    """
    加载临床数据 (Excel)，用于获取测试集受试者的 Age 和 Sex。
    """
    data_dict = {}
    if excel_path and os.path.exists(excel_path):
        print(f"Loading clinical data from {excel_path}...")
        try:
            df = pd.read_excel(excel_path, dtype=str) 
            # 简单的列名匹配逻辑
            id_col = next((c for c in df.columns if 'ID' in c and 'Patient' not in c), df.columns[0])
            age_col = next((c for c in df.columns if 'Age' in c), None)
            sex_col = next((c for c in df.columns if 'Sex' in c), None)
            
            for _, row in df.iterrows():
                # 标准化 ID
                raw_id = str(row[id_col]).strip()
                if raw_id.lower() == 'nan': continue
                pid = raw_id.zfill(3)
                try:
                    age = float(row[age_col])
                    sex = str(row[sex_col]).strip()
                    data_dict[pid] = {'age': age, 'sex': sex}
                except:
                    pass
            return data_dict
        except Exception as e:
            print(f"Error reading Excel: {e}. Falling back to default.")
    return data_dict

# ================= 1. 配置加载 (智能版) =================
def load_config_simple(checkpoint_dir):
    config_atlas_path = os.path.join(checkpoint_dir, "config_atlas.yaml")
    config_data_path = os.path.join(checkpoint_dir, "config_data.yaml")
    
    with open(config_atlas_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(config_data_path, 'r') as f:
        data_config_full = yaml.safe_load(f)
        
    target_dataset = 'mra_atlas'
    if target_dataset in data_config_full:
        config['dataset'] = data_config_full[target_dataset]
    elif 'dataset' in data_config_full:
        config['dataset'] = data_config_full['dataset']
    else:
        config['dataset'] = list(data_config_full.values())[0]
    
    # 强制修正 world_bbox
    if config['dataset'].get('dataset_name') == 'mra_atlas':
        config['dataset']['world_bbox'] = [240, 240, 120]
        
    print(f"Active World BBox: {config['dataset'].get('world_bbox')}")
    return config

# ================= 2. 计算条件均值 Latent (核心逻辑) =================
def get_conditional_mean_latent(args, target_age, target_sex, train_latents, train_dataset, device, gaussian_span=5.0):
    """
    根据目标年龄和性别，计算训练集 Latent 的加权平均值。
    逻辑参考 gen_atlas.py / build_atlas.py
    """
    # 1. 获取训练集的年龄分布
    cond_key = 'scan_age'
    # train_dataset.get_condition_values 返回归一化后的值
    train_ages_norm, _ = train_dataset.get_condition_values(cond_key, normed=True, device=device)
    
    # 2. 归一化目标年龄
    target_age_norm = normalize_condition(args, cond_key, target_age)
    if not isinstance(target_age_norm, torch.Tensor):
        target_age_norm = torch.tensor(target_age_norm).to(device)
    else:
        target_age_norm = target_age_norm.to(device)

    # 3. 计算高斯权重 (基于年龄距离)
    # sigma 计算参考 config
    c_min = args['dataset']['constraints'][cond_key]['min']
    c_max = args['dataset']['constraints'][cond_key]['max']
    c_ratio = 2 / (c_max - c_min) # 因为归一化到了 [-1, 1]
    
    # cond_scale 控制对条件的敏感度
    cond_scale = args['atlas_gen'].get('cond_scale', 0.1)
    sigma = 0.5 * gaussian_span * c_ratio * cond_scale
    
    diff = train_ages_norm - target_age_norm
    weights = torch.exp(-(diff)**2 / (2 * (sigma**2)))
    
    # 4. 根据性别过滤权重 (如果有性别条件)
    if 'sex' in args['dataset']['conditions']:
        # 获取训练集性别 (归一化值)
        train_sex_norm, _ = train_dataset.get_condition_values('sex', normed=True, device=device)
        
        # 归一化目标性别
        target_sex_val = 0.0 if 'male' in target_sex.lower() and 'female' not in target_sex.lower() else 1.0
        target_sex_norm = normalize_condition(args, 'sex', target_sex_val)
        target_sex_norm = torch.tensor(target_sex_norm).to(device)
        
        # 计算性别匹配 mask (完全匹配权重为1，不匹配为0，或者赋予极小权重)
        # 这里使用简单的硬匹配：性别不同权重置 0
        sex_mask = (torch.abs(train_sex_norm - target_sex_norm) < 0.1).float()
        weights = weights * sex_mask

    # 防止除以零
    if torch.sum(weights) == 0:
        print(f"Warning: No matching subjects found for Age={target_age}, Sex={target_sex}. Using uniform mean.")
        weights = torch.ones_like(weights)
        
    weights = weights / torch.sum(weights)
    
    # 5. 加权平均
    # 扩展权重维度以匹配 Latents: (N) -> (N, 1, 1, 1, 1)
    for _ in range(len(train_latents.shape) - 1):
        weights = weights.unsqueeze(-1)
        
    mean_latent = torch.sum(train_latents * weights, dim=0, keepdim=True) # (1, C, H, W, D)
    return mean_latent, target_age_norm

# ================= 3. 数据准备 (Robust Normalization) =================
def prepare_image_data(nii_path, args, device, n_samples=100000):
    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    shape = data.shape
    
    # Robust Normalization
    p01 = np.percentile(data, 1)
    p99 = np.percentile(data, 99)
    data = np.clip(data, p01, p99)
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        data = (data - d_min) / (d_max - d_min)
    
    # 采样
    flat_data = data.flatten()
    threshold = np.percentile(flat_data, 90) 
    fg_indices = np.argwhere(data > threshold)
    bg_indices = np.argwhere(data <= threshold)
    
    n_fg = n_samples // 2
    n_bg = n_samples - n_fg
    
    idx_fg = np.random.choice(len(fg_indices), min(len(fg_indices), n_fg), replace=True) if len(fg_indices) > 0 else []
    sample_fg = fg_indices[idx_fg] if len(idx_fg) > 0 else np.zeros((0, 3))
    
    idx_bg = np.random.choice(len(bg_indices), min(len(bg_indices), n_bg), replace=True) if len(bg_indices) > 0 else []
    sample_bg = bg_indices[idx_bg] if len(idx_bg) > 0 else np.zeros((0, 3))
    
    sample_indices = np.vstack([sample_fg, sample_bg])
    np.random.shuffle(sample_indices)
    
    # 坐标变换
    ones = np.ones((len(sample_indices), 1))
    indices_homo = np.hstack([sample_indices, ones])
    coords_phys = (affine @ indices_homo.T).T[:, :3]
    
    img_center_index = np.array(shape) / 2.0
    center_homo = np.append(img_center_index, 1)
    geometric_center = (affine @ center_homo)[:3]
    coords_centered = coords_phys - geometric_center
    
    world_bbox = np.array(args['dataset']['world_bbox'])
    coords_norm = coords_centered / (world_bbox / 2.0)
    
    values = data[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]]
    values = values[:, None]
    
    return torch.tensor(coords_norm, dtype=torch.float32).to(device), \
           torch.tensor(values, dtype=torch.float32).to(device), \
           affine, shape

# ================= 4. TTA 主流程 =================
def run_dir_tta(input_dir, checkpoint_path, output_dir, excel_path=None, tta_epochs=50, n_samples=100000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config = load_config_simple(checkpoint_dir)
    config['output_dir'] = output_dir
    clinical_data = load_clinical_data(excel_path)
    
    # 初始化模型
    print("Initializing Model...")
    cond_dims = sum([config['dataset']['conditions'][c] for c in config['dataset']['conditions']])
    config['inr_decoder']['cond_dims'] = cond_dims
    model = INR_Decoder(config, device).to(device)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['inr_decoder'])
    
    # [关键步骤] 恢复训练集信息以计算 Mean Latent
    print("Restoring Training Data info for Latent Prior...")
    if 'dataset_df' in checkpoint and 'latents' in checkpoint:
        train_df = checkpoint['dataset_df']
        train_latents = checkpoint['latents'].to(device)
        # 初始化 Dataset 对象以便使用 get_condition_values 等辅助函数
        # 注意 split='train' 确保加载的是训练集逻辑
        train_dataset = Data(config, tsv_file=None, split='train', df_loaded=train_df)
        print(f"Restored Training Set: {len(train_dataset)} subjects.")
    else:
        raise ValueError("Checkpoint missing 'dataset_df' or 'latents'. Cannot compute conditional prior.")

    # 冻结模型权重
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    nii_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    print(f"Found {len(nii_files)} files.")
    
    # 遍历测试集
    for f_idx, nii_path in enumerate(nii_files):
        filename = os.path.basename(nii_path)
        try:
            subject_id = filename.split('.')[0].split('_')[-1]
            if not subject_id.isdigit():
                import re
                match = re.search(r'(\d{3})', filename)
                subject_id = match.group(1) if match else "000"
        except:
            subject_id = "000"
            
        print(f"\n[{f_idx+1}/{len(nii_files)}] Processing {subject_id}...")
        
        # 1. 获取该受试者的 Age 和 Sex
        if subject_id in clinical_data:
            target_age = clinical_data[subject_id]['age']
            target_sex = clinical_data[subject_id]['sex']
            print(f"  > Clinical Info: Age={target_age}, Sex={target_sex}")
        else:
            print(f"  > Warning: No clinical info for {subject_id}, utilizing global mean.")
            target_age = 50.0 # Default fallback
            target_sex = 'M'
            
        # 2. [核心逻辑] 计算初始 Latent Code (Latent Prior)
        # 这就是 Validation 逻辑的核心：利用先验
        z_init, _ = get_conditional_mean_latent(
            config, target_age, target_sex, 
            train_latents, train_dataset, device
        )
        
        # 3. 准备 TTA 变量
        # 复制一份 z_init 并设为可导，作为优化的起点
        latent_z = z_init.clone().detach().requires_grad_(True)
        
        tf_dim = config['inr_decoder']['tf_dim']
        tf_theta = torch.zeros((1, max(tf_dim, 6)), device=device).requires_grad_(True) if tf_dim > 0 else None
        
        # 准备条件向量 (用于 Forward)
        # 注意：这里的条件向量应该与 z_init 对应的条件一致
        # 虽然 z_init 已经包含了条件信息，但显式输入 cond_vector 依然是必要的 (基于 SIREN 结构)
        cond_list = []
        if config['dataset']['conditions'].get('scan_age', False):
            age_norm = normalize_condition(config, 'scan_age', torch.tensor(target_age))
            cond_list.append(age_norm.view(1))
        if config['dataset']['conditions'].get('sex', False):
            sex_val = 0.0 if 'male' in target_sex.lower() and 'female' not in target_sex.lower() else 1.0
            sex_norm = normalize_condition(config, 'sex', torch.tensor(sex_val))
            cond_list.append(sex_norm.view(1))
        
        conditions_vector = torch.cat(cond_list).unsqueeze(0).to(device) if cond_list else torch.zeros((1, 0)).to(device)

        # 4. TTA 优化循环
        optimizer = optim.Adam([
            {'params': latent_z, 'lr': 1e-3},
            {'params': tf_theta, 'lr': 1e-4}
        ]) if tf_theta is not None else optim.Adam([{'params': latent_z, 'lr': 1e-3}])
        
        pbar = tqdm(range(tta_epochs), desc="TTA Optimization", leave=False)
        for epoch in pbar:
            coords, values, _, _ = prepare_image_data(nii_path, config, device, n_samples=n_samples)
            
            optimizer.zero_grad()
            N = coords.shape[0]
            
            # Expand
            tfs_expanded = tf_theta.expand(N, -1) if tf_theta is not None else None
            cond_expanded = conditions_vector.expand(N, -1)
            idcs_df = torch.zeros(N, dtype=torch.long, device=device)
            
            # Forward
            outputs, aux_loss = model(coords, latent_z, cond_expanded, tfs=tfs_expanded, idcs_df=idcs_df)
            pred_mra = outputs[..., 0:1]
            
            loss = F.l1_loss(pred_mra, values)
            if aux_loss is not None:
                loss += 0.01 * aux_loss
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # 5. 生成结果
        print(f"  > Generating output for {subject_id}...")
        ref_nii = nib.load(nii_path)
        ref_affine = ref_nii.affine
        ref_shape = ref_nii.shape
        
        # Grid Generation
        i = torch.arange(0, ref_shape[0], device=device)
        j = torch.arange(0, ref_shape[1], device=device)
        k = torch.arange(0, ref_shape[2], device=device)
        grid = torch.meshgrid(i, j, k, indexing='ij')
        grid_coords_idx = torch.stack(grid, dim=-1).reshape(-1, 3).float()
        
        affine_torch = torch.tensor(ref_affine, dtype=torch.float32, device=device)
        ones = torch.ones((grid_coords_idx.shape[0], 1), device=device)
        grid_homo = torch.cat([grid_coords_idx, ones], dim=1)
        grid_phys = (affine_torch @ grid_homo.T).T[:, :3]
        
        img_center = torch.tensor(ref_shape, device=device) / 2.0
        center_homo = torch.cat([img_center, torch.tensor([1.0], device=device)])
        geo_center = (affine_torch @ center_homo)[:3]
        
        wb_torch = torch.tensor(config['dataset']['world_bbox'], dtype=torch.float32, device=device)
        grid_norm = (grid_phys - geo_center) / (wb_torch / 2.0)
        
        step_size_inference = 100000
        with torch.no_grad():
            output_volume = model.inference(
                grid_norm, latent_z, conditions_vector, list(ref_shape),
                tfs=tf_theta, step_size=step_size_inference
            )
            
            # Save MRA
            mra_rec = output_volume[..., 0].cpu().numpy()
            save_path_mra = os.path.join(output_dir, f"{subject_id}_tta_mra.nii.gz")
            nib.save(nib.Nifti1Image(mra_rec, ref_affine), save_path_mra)
            
            # Save Seg
            sr_dims = sum(config['inr_decoder']['out_dim'][:-1])
            seg_rec = output_volume[..., sr_dims].cpu().numpy()
            save_path_seg = os.path.join(output_dir, f"{subject_id}_tta_seg.nii.gz")
            nib.save(nib.Nifti1Image(seg_rec.astype(np.float32), ref_affine), save_path_seg)
            print(f"  > Saved segmentation to {save_path_seg}")

if __name__ == "__main__":
    INPUT_DIR = "/home/zhangx/TOPCOW/MR_NORM"
    CHECKPOINT_PATH = "/home/zhangx/Cinema_moe/tmp/mra_atlas_20260214_151316_loc/checkpoint_epoch_20.pth"
    OUTPUT_DIR = "/home/zhangx/TOPCOW/tta_results_custom"
    EXCEL_PATH = "/home/zhangx/TOPCOW/2TopCoW_DIZH_Event_Age_Sex_Height_Weight.xlsx" 
    
    run_dir_tta(INPUT_DIR, CHECKPOINT_PATH, OUTPUT_DIR, EXCEL_PATH, tta_epochs=50, n_samples=100000)
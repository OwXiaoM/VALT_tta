import os
import pandas as pd
import yaml
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') 
from tqdm import tqdm
from models.inr_decoder import INR_Decoder
from data_loading.dataset import Data
from utils import generate_world_grid, normalize_condition

def load_config(config_path, data_config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    with open(data_config_path, 'r') as stream:
        data_config_full = yaml.safe_load(stream)
        dataset_name = config.get('config_data', 'mra_atlas')
        if dataset_name not in data_config_full:
             dataset_name = list(data_config_full.keys())[0]
        config['dataset'] = data_config_full[dataset_name]
    
    if isinstance(config['dataset']['subject_ids'], str):
        with open(config['dataset']['subject_ids'], 'r') as stream:
            ids = yaml.safe_load(stream)
            if config['dataset']['dataset_name'] in ids:
                config['dataset']['subject_ids'] = ids[config['dataset']['dataset_name']]['subject_ids']
            else:
                config['dataset']['subject_ids'] = ids['subject_ids']
                
    return config

def get_mean_latent(args, target_age, latents, dataset, device, gaussian_span=5.0):
    cond_key = 'scan_age'
    condition_values, _ = dataset.get_condition_values(cond_key, normed=True, device=device)
    
    target_age_norm = normalize_condition(args, cond_key, target_age)
    if isinstance(target_age_norm, torch.Tensor):
        target_age_norm = target_age_norm.to(device)
    else:
        target_age_norm = torch.tensor(target_age_norm).to(device)

    c_min = args['dataset']['constraints'][cond_key]['min']
    c_max = args['dataset']['constraints'][cond_key]['max']
    c_ratio = 2 / (c_max - c_min)
    
    sigma = 0.5 * gaussian_span * c_ratio * args['atlas_gen']['cond_scale']
    
    diff = condition_values - target_age_norm
    weights = torch.exp(-(diff)**2 / (2 * (sigma**2)))
    weights = weights / torch.sum(weights)
    
    for _ in range(len(latents.shape) - 1):
        weights = weights.unsqueeze(-1)
        
    mean_latent = torch.sum(latents * weights, dim=0, keepdim=True)
    return mean_latent, target_age_norm

def main():
    # ================= 配置区域 =================
    checkpoint_path = "/home/zhangx/Cinema_moe/tmp/mra_atlas_20260211_210406_loc/checkpoint_epoch_12.pth"
    config_atlas_path = "/home/zhangx/Cinema_moe/tmp/mra_atlas_20260211_210406_loc/config_atlas.yaml"
    config_data_path = "/home/zhangx/CINeMA/tmp/mra_atlas_20260130_134905_loc/config_data.yaml"
    target_ages = [20, 30, 40, 50, 60, 70, 80]
    output_dir = "/home/zhangx/CINeMA/tmp/mra_atlas_20260130_134905_loc/vessel_atlas"
    # ===========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # 2. 加载配置
    print("Loading configuration...")
    config = load_config(config_atlas_path, config_data_path)
    config['output_dir'] = output_dir # 修复直方图保存路径

    # 3. 初始化数据集
    print("Initializing dataset...")
    config['batch_size'] = 1 
    
    tsv_path = config['dataset']['tsv_file']
    print(f"Reading TSV file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    dataset = Data(config, df, split='train')
    
    # 4. 初始化模型
    print("Initializing INR Decoder...")
    config['inr_decoder']['cond_dims'] = sum([config['dataset']['conditions'][c] for c in config['dataset']['conditions']])
    model = INR_Decoder(config, device).to(device)
    model.eval()

    # 5. 加载权重 [关键修复点]
    print(f"Loading checkpoint: {checkpoint_path}")
    # 添加 weights_only=False 以允许加载 Pandas DataFrame 等复杂对象
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['inr_decoder'])
    train_latents = checkpoint['latents'].to(device)
    print(f"Loaded {len(train_latents)} subject latents.")

    # 6. 准备网格
    grid_coords, grid_shape, affine = generate_world_grid(config, device=device)
    
    # 7. 生成循环
    vessel_label_idx = 1
    if 'Vessel' in config['dataset']['label_names']:
        vessel_label_idx = config['dataset']['label_names'].index('Vessel')
    
    has_sex_cond = config['dataset']['conditions'].get('sex', False)
    sex_values = ['M', 'F'] if has_sex_cond else ['All']

    print("Starting Atlas Generation...")
    
    with torch.no_grad():
        for age in tqdm(target_ages, desc="Generating Ages"):
            mean_latent, age_norm = get_mean_latent(config, age, train_latents, dataset, device)
            
            for sex in sex_values:
                cond_list = []
                if config['dataset']['conditions'].get('scan_age', False):
                    cond_list.append(age_norm)
                
                if has_sex_cond:
                    sex_idx = 0.0 if sex == 'M' else 1.0
                    sex_norm = normalize_condition(config, 'sex', sex_idx)
                    cond_list.append(torch.tensor(sex_norm).to(device))
                
                if len(cond_list) > 0:
                    cond_vector = torch.stack(cond_list).float().unsqueeze(0)
                else:
                    cond_vector = torch.zeros((1, 0)).to(device)

                output_volume = model.inference(
                    grid_coords, 
                    mean_latent, 
                    cond_vector, 
                    grid_shape, 
                    tfs=None, 
                    step_size=50000
                )
                
                sr_dims = sum(config['inr_decoder']['out_dim'][:-1])
                soft_start_idx = sr_dims + 1 
                target_channel_idx = soft_start_idx + vessel_label_idx
                
                vessel_prob_map = output_volume[..., target_channel_idx].cpu().numpy()
                
                filename = f"Atlas_Age-{age}_Sex-{sex}_VesselProb.nii.gz"
                save_path = os.path.join(output_dir, filename)
                
                if isinstance(affine, torch.Tensor):
                    affine = affine.cpu().numpy()
                    
                nii = nib.Nifti1Image(vessel_prob_map.astype(np.float32), affine)
                nib.save(nii, save_path)
                
    print(f"Done! Atlases saved to {output_dir}")

if __name__ == "__main__":
    main()
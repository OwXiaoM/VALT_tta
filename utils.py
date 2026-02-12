import math
import json
from types import SimpleNamespace
import ants
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import nibabel as nib
import wandb as wd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import scipy.ndimage as ndi
import torchio as tio

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def dict_to_simplenamespace(d):
    """ Recursively converts dictionary to SimpleNamespace. """
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    else:
        return d


class MaskSubjectTransform(tio.Transform):
    """
    A custom TorchIO transform that uses a segmentation image 
    to mask the intensity images in a subject.
    """
    def __init__(self, segmentation_key, intensity_keys=None, **kwargs):
        super().__init__(**kwargs)
        self.segmentation_key = segmentation_key
        self.intensity_keys = intensity_keys

    def apply_transform(self, subject):
        seg_image = subject[self.segmentation_key]
        seg_data = seg_image.data  # shape: (1, X, Y, Z)
        
        if self.intensity_keys is None:
            keys_to_mask = [k for k in subject.get_images_names() if k != self.segmentation_key]
        else:
            keys_to_mask = self.intensity_keys

        for key in keys_to_mask:
            image = subject[key]
            image.data = image.data * (seg_data > 0)
        
        return subject


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tf_weight = args['optimizer']['tf_weight']
        self.moe_weight = args['optimizer'].get('moe_weight', 0.1) 
        
        self.n_classes = args['inr_decoder']['out_dim'][-1]-1
        self.sr_dims = sum(args['inr_decoder']['out_dim'][:-1])
        self.criterion_sr = nn.MSELoss() if args['optimizer']['loss_metric'] == 'mse' else nn.L1Loss()
        self.ce_weights = torch.tensor(args['dataset']['class_weights'], dtype=torch.float32, device=args['device']) if args['dataset']['class_weights'] is not None else None
        self.criterion_seg = nn.CrossEntropyLoss(weight=self.ce_weights)

    def forward(self, output, target, tfs, moe_loss=None, sr_weight=1.0, seg_weight=1.0):
        loss = {'seg': torch.tensor(0.0), 
                'sr': sr_weight * self.criterion_sr(output[..., :self.sr_dims], target[..., :self.sr_dims]),
                'trafo': torch.tensor(0.0), 
                'moe': torch.tensor(0.0),
                'total': 0.0}

        if seg_weight > 0:
            loss['seg'] = seg_weight * self.criterion_seg(output[..., self.sr_dims:], target[..., -1].to(torch.int64))
            
        if tfs is not None:
            loss['tf_rot'] = torch.mean(tfs[..., :3] ** 2)
            loss['tf_trans'] = torch.mean(tfs[..., 3:6] ** 2)
            loss['tf_scale'] = torch.mean(tfs[..., 6:9] ** 2) if tfs.shape[-1] == 9 else torch.tensor(0.0)
            loss['tf'] = loss['tf_rot'] + loss['tf_trans'] + loss['tf_scale']
        
        if moe_loss is not None and self.moe_weight > 0:
            loss['moe'] = self.moe_weight * moe_loss

        loss['total'] = loss['sr'] + seg_weight * loss['seg'] + self.tf_weight * loss['tf'] + loss['moe']
        return loss


def scale_affine(affine, new_spacing, bbox):
    old_spacing = torch.sqrt(torch.sum(affine[:3, :3] ** 2, axis=0))
    scale = torch.tensor(new_spacing) / old_spacing
    bbox_size = (bbox['max'] - bbox['min'])
    shape = (torch.ceil_(bbox_size / scale).int())
    affine[:3, :3] = torch.matmul(affine[:3, :3], torch.diag(scale))
    affine[:3, 3] = affine[:3, 3] + (bbox['min'] * old_spacing)
    return affine, shape


def compute_ncc(prediction, reference):
    mean_pred = np.mean(prediction)
    mean_ref = np.mean(reference)
    numerator = np.sum((prediction - mean_pred) * (reference - mean_ref))
    denominator = np.sqrt(np.sum((prediction - mean_pred) ** 2) * np.sum((reference - mean_ref) ** 2))
    ncc = numerator / denominator
    return ncc.astype(np.float64)


def embed2affine(embed):
    R = euler2rot(embed[..., :3])
    t = embed[..., 3:6]
    if embed.shape[-1] >= 9:
        S = torch.diag_embed(1.0 + embed[..., 6:9])
        R = torch.matmul(R, S)
    if embed.shape[-1] == 12:
        S_x = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_y = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_z = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_x[..., 0, 1] = embed[..., 10]
        S_x[..., 0, 2] = embed[..., 11]
        S_y[..., 1, 0] = embed[..., 9]
        S_y[..., 1, 2] = embed[..., 11]
        S_z[..., 2, 0] = embed[..., 9]
        S_z[..., 2, 1] = embed[..., 10]
        R = torch.matmul(R, S_x)
        R = torch.matmul(R, S_y)
        R = torch.matmul(R, S_z)
    return R, t


def euler2rot(theta):
    c1 = torch.cos(theta[..., 0])
    s1 = torch.sin(theta[..., 0])
    c2 = torch.cos(theta[..., 1])
    s2 = torch.sin(theta[..., 1])
    c3 = torch.cos(theta[..., 2])
    s3 = torch.sin(theta[..., 2])
    r11 = c1*c3 - c2*s1*s3
    r12 = -c1*s3 - c2*c3*s1
    r13 = s1*s2
    r21 = c3*s1 + c1*c2*s3
    r22 = c1*c2*c3 - s1*s3
    r23 = -c1*s2
    r31 = s2*s3
    r32 = c3*s2
    r33 = c2
    R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1)
    R = R.view(R.shape[:-1] + (3, 3))
    return R


def harmonize_labels(subject_seg, dataset):
    if 'dhcp_neo' in dataset:
        subject_seg[subject_seg == 4] = 0
        subject_seg[subject_seg == 8] = 6
        subject_seg[subject_seg == 9] = 7
    elif 'dhcp_fetal' in dataset or 'marsfet' in dataset:
        subject_seg[subject_seg == 4] = 0
        subject_seg[subject_seg == 5] = 4
        subject_seg[subject_seg == 3] = 3
        subject_seg[subject_seg == 6] = 3
        subject_seg[subject_seg == 7] = 3
        subject_seg[subject_seg == 8] = 3
        subject_seg[subject_seg == 9] = 3
    else:
        raise NotImplementedError("Atlas Type not implemented.")
    return subject_seg


def to_device(x, device='cuda'):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x 
    

def normalize_condition(args, condition_key, condition_values, cond_scale=None):
    c_scale = args['atlas_gen']['cond_scale'] if cond_scale is None else cond_scale
    constraint = args['dataset']['constraints'][condition_key]
    
    if constraint.get('type') == 'categoric':
        c_min = 0.0
        c_max = float(len(constraint['values']) - 1)
    else:
        c_min = constraint['min']
        c_max = constraint['max']

    if c_max > c_min:
        cv = 2 * ((condition_values - c_min) / (c_max - c_min) - 0.5)
    else:
        cv = 0.0
        
    cv = cv * c_scale
    return cv


def generate_combinations(args_data, conditions, keys=None, idx=0, current=None, results=None):
    if conditions is None: 
        return [[]]
    if keys is None:
        keys = list(conditions.keys())
    if current is None:
        current = []
    if results is None:
        results = []

    key = keys[idx]
    values = conditions[key]['values']

    for value in values:
        if not conditions[key]['normed_values']:
            value = normalize_condition(args_data, key, value)
        else:
            value = value * args_data['atlas_gen']['cond_scale']
        next_current = current + [value]
        if idx == len(keys) - 1:
            results.append(next_current)
        else:
            generate_combinations(args_data, conditions, keys, idx + 1, next_current, results)

    return results


def generate_world_grid(args, normed=True, device='cpu'):
    world_bbox = args['dataset']['world_bbox']
    spacing = args['atlas_gen']['spacing']
    
    x = torch.arange(0, world_bbox[0], spacing[0], device=device)
    y = torch.arange(0, world_bbox[1], spacing[1], device=device)
    z = torch.arange(0, world_bbox[2], spacing[2], device=device)
    
    if normed:
        x = x / world_bbox[0] * 2 - 1
        y = y / world_bbox[1] * 2 - 1
        z = z / world_bbox[2] * 2 - 1
        
    grid = torch.meshgrid(x, y, z, indexing='ij')
    grid_shape = list(grid[0].shape)
    grid_coords = torch.stack(grid, dim=-1).reshape(-1, 3)
    
    affine = torch.diag(torch.tensor([spacing[0], spacing[1], spacing[2], 1.0], device=device))
    origin_offset = -torch.tensor(world_bbox, device=device) / 2.0
    affine[:3, 3] = origin_offset

    return grid_coords, grid_shape, affine


def save_atlas(args, atlas, affine, temp_steps, condition_vectors, epoch):
    x, y, z, num_mods, n_conds, t = atlas.shape
    if isinstance(atlas, torch.Tensor):
        try:
            atlas = atlas.detach().cpu().numpy()
        except:
            atlas = atlas.numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    mod_labels = args['dataset']['modalities']
    if args['save_certainty_maps']:
        seg_labels = [
            f"CertaintyMaps/{label}" 
            for label in args['dataset']['label_names'] 
            if label.lower() not in ['Background']
        ]
        mod_labels = mod_labels + seg_labels
    for c in range(n_conds):
        for i in range(len(mod_labels)):
            filename = (f'{mod_labels[i]}={temp_steps[0]}-{temp_steps[-1]}_cond={c}_ep={epoch}.nii.gz')
            save_img(atlas[..., i, c, :], affine=affine, output_path=args['output_dir'], filename=filename)
    print('Atlas saved to {}'.format(args['output_dir']))


def typecheck_img_affine(img, affine):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if img.dtype == 'int64' or img.dtype == 'int32':
        img = img.astype(np.int16)
    elif img.dtype == 'float64':
        img = img.astype(np.float32)
    return img, affine


def save_subject(args, img, affine, sub_row_df=None, sub_name=None, epoch=0, split='train'):
    if sub_row_df is not None:
        sub_name = sub_row_df['subject_id']
    elif sub_name is None:
        print("No subject name provided, assigning random subject name.")
        sub_name = 'subject_' + str(np.random.randint(100000))

    img, affine = typecheck_img_affine(img, affine)
    mytx = None
    modalities = args['dataset']['modalities']
    for i, mod in enumerate(modalities):
        if i == len(modalities)-1 and 'Seg' in mod:
            img_mod = img[..., i].astype(np.int16)
        else:
            img_mod = img[..., i].astype(np.float32)
        filename = f'{split}/{mod}_{sub_name}_ep={epoch}.nii.gz'
        save_img(img_mod, affine, args['output_dir'], filename)
        if epoch == 0 and sub_row_df is not None:
            is_seg = (i == len(modalities)-1 and 'Seg' in mod)
            pred_mod_nii = nib.Nifti1Image(img_mod, affine)
            pred_mask_nii = nib.Nifti1Image(img[..., -1], affine)
            ref_mod_nii = nib.load(sub_row_df[mod])
            pred_mod_nii, ref_mod_nii, mytx = reg_imgs(pred_mod_nii, ref_mod_nii, pred_mask_nii, mytx, 'Rigid', is_seg)
            filename_ref = f'{split}/{mod}_{sub_name}_ref.nii.gz'
            save_img(ref_mod_nii.get_fdata(), ref_mod_nii.affine, args['output_dir'], filename_ref)


def save_img(img, affine, output_path, filename):
    if img.dtype == 'int64' or img.dtype == 'int32':
        img = img.astype(np.int16)
    elif img.dtype == 'float64':
        img = img.astype(np.float32)
    full_path = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    img_nii = nib.Nifti1Image(img, affine)
    nib.save(img_nii, full_path)
    print(f'Saved {filename} to {output_path}')


def assert_correct_coord_normalization(coords, min_val=-1.0, max_val=1.0):
    min_x, min_y, min_z = coords.min(axis=0)
    max_x, max_y, max_z = coords.max(axis=0)
    assert min_x >= min_val, f"min_x = {min_x} is less than min_val = {min_val}"
    assert max_x <= max_val, f"max_x = {max_x} is greater than max_val = {max_val}"
    assert min_y >= min_val, f"min_y = {min_y} is less than min_val = {min_val}"
    assert max_y <= max_val, f"max_y = {max_y} is greater than max_val = {max_val}"
    assert min_z >= min_val, f"min_z = {min_z} is less than min_val = {min_val}"
    assert max_z <= max_val, f"max_z = {max_z} is greater than max_val = {max_val}"

def squeeze_modalities(modality, path, write2disk=False):
    if modality.ndim == 4:
        mod_np = modality.get_fdata()
        mod_np = np.squeeze(mod_np)
        mod_nii = nib.Nifti1Image(mod_np, modality.affine)
        if write2disk:
            mod_nii.to_filename(path)
        return mod_nii
    else:
        return modality

def add_background_halo(label_names, seg_nii, halo_width=1.5, background_label_str='Background'):
    seg = seg_nii.get_fdata()
    bg_label = label_names.index(background_label_str)
    mask_bg = (seg>0).astype(np.float32)
    mask_bg = (ndi.gaussian_filter(mask_bg, sigma=halo_width) > 0.001).astype(np.uint8) * bg_label
    mask_bg[seg > 0] = seg[seg > 0]
    return nib.Nifti1Image(mask_bg, seg_nii.affine)


def mask_nifti(nii, mask):
    data = nii.get_fdata()
    data *= mask
    return nib.Nifti1Image(data, nii.affine)

# ================= [新增] 替换 ants.from_nibabel =================
def nib_to_ants(nii_img):
    """
    Manually convert Nibabel image to ANTs image, bypassing ants.from_nibabel
    which might be missing in older antspy versions.
    """
    # 1. Get Data
    data = nii_img.get_fdata().astype(np.float32)
    
    # 2. Get Affine
    affine = nii_img.affine
    
    # 3. Decompose Affine
    # Spacing
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    # Origin
    origin = affine[:3, 3]
    # Direction (Normalized Rotation Matrix)
    # Avoid division by zero
    spacing_nonzero = spacing.copy()
    spacing_nonzero[spacing_nonzero == 0] = 1.0
    direction = affine[:3, :3] @ np.diag(1/spacing_nonzero)
    
    # 4. Create ANTs Image
    # Note: Nibabel is (x,y,z), ANTs is (x,y,z). 
    # Warning: Direction handling can be tricky (RAS vs LPS). 
    # Usually passing direction matrix explicitly preserves physical alignment.
    img_ants = ants.from_numpy(data=data, 
                               origin=origin.tolist(), 
                               spacing=spacing.tolist(), 
                               direction=direction)
    return img_ants
# ===============================================================

def compute_metrics(args, pred, affine, df_row_dict, epoch, split, reg_type='Rigid', bg_label=None):
    """
    Compute metrics for a predicted subject.
    Updated to use nib_to_ants instead of ants.from_nibabel.
    """
    pred, affine = typecheck_img_affine(pred, affine)
    metrics = {'Subject': df_row_dict['subject_id'], 'PSNR': [], 'SSIM': [], 'DICE': []}
    modalities = args['dataset']['modalities']
    sub_id = df_row_dict['subject_id']

    if bg_label is None and 'Background' in args['dataset']['label_names']:
        bg_label = args['dataset']['label_names'].index('Background')
    else:
        bg_label = 0 

    for i, mod in enumerate(modalities):
        is_seg = (i == len(modalities)-1 and ('Seg' in mod or 'Mask' in mod))
        
        mod_ref_nii = nib.load(df_row_dict[mod])
        mod_pred_nii = nib.Nifti1Image(pred[..., i], affine)

        # [修改] 使用 nib_to_ants
        try:
            fix = nib_to_ants(mod_ref_nii) # GT
            mov = nib_to_ants(mod_pred_nii) # Pred
        except Exception as e:
            print(f"[Error] Failed to convert nifti to ants for {mod}: {e}. Skipping.")
            continue

        try:
            interp_type = 'genericLabel' if is_seg else 'linear'
            # 重采样 Pred (Mov) 到 GT (Fix) 的网格
            mov = ants.resample_image_to_target(mov, fix, interp_type=interp_type)
        except Exception as e:
            print(f"[Error] Resampling failed for {mod}: {e}. Skipping metric.")
            continue

        if not is_seg:
             mov_arr = mov.numpy()
             fix_arr = fix.numpy()
             if mov_arr.max() - mov_arr.min() > 1e-6:
                 mov = (mov - mov_arr.min()) / (mov_arr.max() - mov_arr.min())
             if fix_arr.max() - fix_arr.min() > 1e-6:
                 fix = (fix - fix_arr.min()) / (fix_arr.max() - fix_arr.min())

        mod_pred_reg_nii = ants.to_nibabel(mov)
        mod_ref_nii = ants.to_nibabel(fix)
        
        bbox, _ = get_bbox(mod_ref_nii) 
        x_min, y_min, z_min = bbox[0]
        x_max, y_max, z_max = bbox[1]+1
        
        mod_pred_reg_cropped = mod_pred_reg_nii.get_fdata()[x_min:x_max, y_min:y_max, z_min:z_max]
        mod_ref_cropped = mod_ref_nii.get_fdata()[x_min:x_max, y_min:y_max, z_min:z_max]
        
        if not is_seg:
            metrics['PSNR'].append(psnr_metric(mod_pred_reg_cropped, mod_ref_cropped, data_range=1.0))
            metrics['SSIM'].append(ssim_metric(mod_pred_reg_cropped, mod_ref_cropped, data_range=1.0))
        else:
            metrics['DICE'].append(compute_dice(mod_pred_reg_cropped, mod_ref_cropped, bg_label))
        
        if args['save_imgs'][split]:
            fn_pred = f'{split}/{mod}_{sub_id}_ep={epoch}.nii.gz'
            fn_ref = f'{split}/{mod}_{sub_id}_ref.nii.gz'
            save_img(mod_pred_reg_nii.get_fdata(), mod_pred_reg_nii.affine, args['output_dir'], fn_pred)
            save_img(mod_ref_nii.get_fdata(), mod_ref_nii.affine, args['output_dir'], fn_ref)

    return metrics
    
def reg_imgs(fix_nii, mov_nii, mask_mov_nii=None, mytx=None, reg_type='Rigid', is_seg=False):
    """
    Updated to use nib_to_ants.
    """
    # 辅助函数：创建全零图（保底）
    def get_zeros_like_fixed(fixed_nii):
        zeros_data = np.zeros(fixed_nii.shape, dtype=fixed_nii.get_fdata().dtype)
        return nib.Nifti1Image(zeros_data, fixed_nii.affine, fixed_nii.header)

    try:
        # [修改] 使用 nib_to_ants
        fix = nib_to_ants(fix_nii)
        mov = nib_to_ants(mov_nii)
    except Exception as e:
        print(f"[Error] Failed to convert nifti to ants: {e}")
        return fix_nii, get_zeros_like_fixed(fix_nii), None

    # Mask 处理
    mask_mov = None
    if mask_mov_nii is not None:
        try:
             mask_mov = nib_to_ants(mask_mov_nii) > 0
        except:
             pass

    ip = 'genericLabel' if is_seg else 'linear'
    fallback = False

    if np.sum(np.abs(fix.numpy())) < 1e-3:
        print("[Warning] Fixed image (Pred) is empty. Registration skipped. Using Identity Resample.")
        fallback = True
    elif np.sum(np.abs(mov.numpy())) < 1e-3:
        print("[Warning] Moving image (Ref) is empty.")
        return fix_nii, get_zeros_like_fixed(fix_nii), None

    if not fallback and mytx is None:
        if mask_mov is not None:
            mov = mov * mask_mov 
        try:
            mytx = ants.registration(fix, mov, type_of_transform=reg_type)
            mov_reg = mytx['warpedmovout']
        except Exception as e:
            print(f"[Warning] ANTs registration failed: {e}. Falling back to Identity Transform.")
            fallback = True
    
    if fallback:
        try:
            mov_reg = ants.resample_image_to_target(mov, fix, interp_type=ip)
            mytx = None
        except Exception as e:
            print(f"[Error] Identity resample failed: {e}. Returning zeros.")
            return fix_nii, get_zeros_like_fixed(fix_nii), None

    elif mytx is not None:
        try:
            if 'fwdtransforms' in mytx:
                mov_reg = ants.apply_transforms(fix, mov, mytx['fwdtransforms'], interpolator=ip)
            else:
                pass 
        except Exception as e:
             print(f"[Error] Apply transforms failed: {e}. Returning zeros.")
             return fix_nii, get_zeros_like_fixed(fix_nii), None

    if mask_mov is not None and mytx is not None and not fallback: 
        try:
            mask_mov_reg = ants.apply_transforms(fix, mask_mov, mytx['fwdtransforms'], interpolator='genericLabel')
            fix = fix * mask_mov_reg
        except:
            pass

    if not is_seg: 
        mov_arr = mov_reg.numpy()
        mov_min, mov_max = mov_arr.min(), mov_arr.max()
        if (mov_max - mov_min) > 1e-6:
            mov_reg = (mov_reg - mov_min) / (mov_max - mov_min)

    fix_nii_out = ants.to_nibabel(fix)
    mov_reg_nii_out = ants.to_nibabel(mov_reg)
    
    return fix_nii_out, mov_reg_nii_out, mytx

def get_bbox(img):
    if isinstance(img, nib.Nifti1Image):
        img = img.get_fdata()
    
    non_zero_voxels = np.argwhere(img != 0)
    
    if non_zero_voxels.size == 0:
        shape = img.shape
        min_coords = np.array([0, 0, 0])
        max_coords = np.array([shape[0]-1, shape[1]-1, shape[2]-1])
        return np.stack([min_coords, max_coords], axis=0), img

    min_coords = non_zero_voxels.min(axis=0)
    max_coords = non_zero_voxels.max(axis=0)
    img_cropped = img[min_coords[0]:max_coords[0]+1, 
                      min_coords[1]:max_coords[1]+1, 
                      min_coords[2]:max_coords[2]+1]
    return np.stack([min_coords, max_coords], axis=0), img_cropped

def compute_dice(pred, ref, bg_label):
    if pred.shape != ref.shape:
        raise ValueError("The shape of pred and ref must be the same.")
    
    labels = np.union1d(np.unique(pred), np.unique(ref))
    labels = labels[(labels != 0) & (labels != bg_label)]
    
    if len(labels) == 0:
        return 1.0
    
    dice_scores = []
    for label in labels:
        pred_mask = (pred == label)
        ref_mask = (ref == label)
        intersection = np.count_nonzero(pred_mask & ref_mask)
        pred_sum = np.count_nonzero(pred_mask)
        ref_sum = np.count_nonzero(ref_mask)
        
        if pred_sum + ref_sum == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / (pred_sum + ref_sum)
        dice_scores.append(dice)
    
    return np.mean(dice_scores)

def log_metrics(args, metrics, epoch, df=None, split='train'):
    metrics_keys = list(metrics[0].keys())
    mod_keys = args['dataset']['modalities']
    for metric_key in metrics_keys:
        if metric_key == 'Subject': continue
        metric_vals = [m[metric_key][0] for m in metrics]
        metric_vals = np.array(metric_vals) 
        metric_mean = np.mean(metric_vals, axis=0).item()
        metric_std = np.std(metric_vals, axis=0).item()
        print(f"{metric_key}: {metric_mean:.3f} +/- {metric_std:.3f}")
        if args['logging']: wd.log({f"{split}/{metric_key}": metric_mean})

    with open(os.path.join(args['output_dir'], f'{split}/{split}_metrics_ep={epoch}.json'), 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
    if df is not None:
        df.to_csv(os.path.join(args['output_dir'], f'{split}/{split}_df.csv'), index=False)


def log_loss(loss, epoch, split, log=True):
    if log: 
        wd.log({f"{split}/loss": loss['total'].item()})
        wd.log({f"{split}/loss_sr": loss['sr'].item()})
        wd.log({f"{split}/loss_seg": loss['seg'].item()})
        wd.log({f"{split}/loss_moe": loss['moe'].item()})
        wd.log({f"{split}/loss_tf_rot": loss['tf_rot'].item()})
        wd.log({f"{split}/loss_tf_trans": loss['tf_trans'].item()})
        wd.log({f"{split}/loss_tf_scale": loss['tf_scale'].item()})
        wd.log({f"{split}/loss_tf": loss['tf'].item()})
        wd.log({f"{split}/epoch": epoch})


def normalize_intensities(values, norm_type):
    values = np.clip(values, 0, None)
    values_mod = values[..., :-1]
    if norm_type == 'minmax':
        v_min, v_max = values_mod.min(axis=0), values_mod.max(axis=0)
        values_mod = (values_mod - v_min) / (v_max - v_min)
    elif norm_type == 'zscore':
        v_mean, v_std = values_mod.mean(axis=0), values_mod.std(axis=0)
        values_mod = (values_mod - v_mean) / (v_std + 1e-5)
    values[..., :-1] = values_mod
    return values


def denormalize_conditions(args, cond_key, values):
    constraint = args['dataset']['constraints'][cond_key]
    
    if constraint.get('type') == 'categoric':
        c_min = 0.0
        c_max = float(len(constraint['values']) - 1)
    else:
        c_min = constraint['min']
        c_max = constraint['max']
        
    c_scale = args['atlas_gen']['cond_scale']
    
    if c_max > c_min:
        values = (values / c_scale + 1) / 2 * (c_max - c_min) + c_min
    else:
        values = values * 0 + c_min
        
    return values
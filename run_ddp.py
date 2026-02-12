import os
import sys
import yaml
import argparse
import wandb as wd
import torch
import torch.distributed as dist
from datetime import datetime

# 引入 DDP Builder
from build_atlas_ddp import AtlasBuilderDDP  

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def initial_setup(cmd_args=None):
    rank = cmd_args.get('rank', 0)
    
    with open('./configs/config_atlas.yaml', 'r') as stream:
        args_atlas = yaml.safe_load(stream)
    with open('./configs/config_data.yaml', 'r') as stream:
        config_data = cmd_args['config_data'] if 'config_data' in cmd_args else args_atlas['config_data']
        args_data = {'dataset': yaml.safe_load(stream)[config_data]}
    args = {**args_data, **args_atlas}
    with open(args['dataset']['subject_ids'], 'r') as stream:
        args['dataset']['subject_ids'] = yaml.safe_load(stream)[args['dataset']['dataset_name']]['subject_ids']
    if cmd_args is not None:
        args = override_args(args, cmd_args)

    # 只在 Rank 0 创建目录和 wandb
    if rank == 0:
        job_id = os.getenv("SLURM_JOB_ID", "loc")[-3:]
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dsetup = args['config_data']
        run_name =f"{dsetup}_{time_stamp}_{job_id}"
        args['output_dir'] = f"{args['output_dir']}/{run_name}"
        os.makedirs(args['output_dir'], exist_ok=True)
        print(f"Output directory: {args['output_dir']}")

        with open(os.path.join(args['output_dir'], 'config_data.yaml'), 'w') as f:
            yaml.dump(args_data, f)
        with open(os.path.join(args['output_dir'], 'config_atlas.yaml'), 'w') as f:
            yaml.dump(args_atlas, f)
        
        if args['logging']: 
            wd.init(config=args, project=args['project_name'], 
                    entity=args['wandb_entity'], name=run_name)
    
    return args

def override_args(config_args, cmd_args):
    for key, value in cmd_args.items():
        if key in ['rank', 'local_rank', 'world_size', 'is_distributed']: continue 
        key1, key2 = key.split("__") if "__" in key else (key, None)
        if key2 is None:
            if value is not None:
                config_args[key] = value
        else:
            if value is not None:
                config_args[key1][key2] = value
    return config_args

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="CINeMA Atlas Builder DDP")
    parser.add_argument("--config_data", type=str, help="Configuration data")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--local-rank", type=int, default=int(os.getenv("LOCAL_RANK", -1)))
    # 支持 torchrun 传递的参数
    args, unknown = parser.parse_known_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    return cmd_args

def main():
    # 初始化 DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        print("Not running in Distributed Mode. Fallback to single GPU.")
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cmd_args = parse_cmd_args()
    cmd_args['rank'] = rank
    cmd_args['local_rank'] = local_rank
    cmd_args['world_size'] = world_size
    
    args = initial_setup(cmd_args)
    
    # 注入环境参数
    args['device'] = device
    args['rank'] = rank
    args['local_rank'] = local_rank
    args['world_size'] = world_size

    # Logger 只在 Rank 0 开启
    if rank == 0:
        log_dir = os.path.join(args['output_dir'], 'train')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training_log.txt')
        sys.stdout = Logger(log_file)
        sys.stderr = sys.stdout 
        print(f"Logging initialized. All outputs will be saved to {log_file}")
        print(f"DDP Enabled. World Size: {world_size}")

    atlas_builder = AtlasBuilderDDP(args)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
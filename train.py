from data.synthDataset import get_synthetic_dataset
from models import training
import argparse
import torch
import json, os, yaml
import torch
from copy import deepcopy

from models.NePu import get_encoder, get_decoder, get_renderer

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
#TODO incorporate in yaml
parser.add_argument('-interp_dec', action='store_true')
parser.set_defaults(interp_dec=False)
parser.add_argument('-pointnet_enc', action='store_true')
parser.set_defaults(pointnet_enc=False)

parser.add_argument('-pc_samples' , default=300, type=int)
parser.add_argument('-batch_size' , default=64, type=int)
parser.add_argument('-cuda_device', default=0, type=int)
parser.add_argument('-data', required=True, type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))

fname_data_cfg = './configs/data_configs.yaml'
with open(fname_data_cfg, 'r') as f:
    data_cfg = yaml.safe_load(f)

CFG['data']['nkps'] = data_cfg['nkps'][args.data]
CFG['data']['ncams'] = data_cfg['ncams'][args.data]
CFG['data']['radius'] = data_cfg['radii'][args.data]
CFG['data']['num_datapoints'] = data_cfg['num_datapoints'][args.data]

exp_dir = './experiments/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
if not os.path.exists(exp_dir):
    print('Creating checkpoint dir: ' + exp_dir)
    os.makedirs(exp_dir)
    with open(fname, 'w') as yaml_file:
        yaml.safe_dump(CFG, yaml_file, default_flow_style=False)
else:
    with open(fname, 'r') as f:
        print('Loading config file from: ' + fname)
        CFG = yaml.safe_load(f)

#TODO print in nicer format!!
print(CFG)

torch.cuda.set_device(args.cuda_device)

train_dataset = get_synthetic_dataset(args.data, 'train', 'training', CFG)
val_dataset = get_synthetic_dataset(args.data, 'val', 'training', CFG)

device = torch.device("cuda")

encoder = get_encoder(CFG)
decoder = get_decoder(CFG)
renderer = get_renderer(CFG)

train_cfg = CFG['training']
trainer = training.Trainer(encoder, decoder, renderer, train_cfg, device, train_dataset, val_dataset, args.exp_name,
                                optimizer='Adam')
trainer.train_model(2000)

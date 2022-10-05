import torch
import numpy as np
import argparse, yaml

from models.NePu import get_encoder, get_decoder,  get_renderer
from data.synthDataset import get_synthetic_dataset


parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)
parser.add_argument('-data', required=True, type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

exp_dir = './experiments/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
with open(fname, 'r') as f:
    print('Loading config file from: ' + fname)
    CFG = yaml.safe_load(f)

print(CFG)

device = torch.device("cuda")

CFG['training']['npoints_decoder'] = 10
dataset = get_synthetic_dataset(args.data, 'train', 'uniform', CFG)


encoder = get_encoder(CFG).float()

encoder = encoder.to(device)
encoder.float()
encoder.eval()

checkpoint = torch.load(exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint), map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])


zs = []
for ind in range(len(dataset)):
    data = dataset.__getitem__(ind)
    gt_pos = data.get('input_pos').to(device).unsqueeze(0).contiguous()
    inp_feats = data.get('input_feats').to(device).unsqueeze(0)

    z = encoder(gt_pos, inp_feats)
    zs.append(z.detach().cpu().numpy())
    print('Done encoding example: {}'.format(ind))

zs = np.stack(zs, axis=0)

np.save(exp_dir + '/all_z_train_{}.npy'.format(args.checkpoint), zs)




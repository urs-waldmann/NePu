import argparse
import os, yaml
import torch
import numpy as np

from data.synthDataset import get_synthetic_dataset
from models.NePu import get_encoder, get_decoder, get_renderer
from models.utils import get2Dkps


parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)
parser.add_argument('-data', required=True, type=str)
parser.add_argument('-res_factor', type=float, default=1.0)
parser.add_argument('-npixel_per_batch', type=int, default=50000)
parser.add_argument('-view_cfg_path', type=str)

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

radius = CFG['data']['radius']
CAMS = list(range(CFG['data']['ncams']))
nkps = CFG['data']['nkps']

encoder = get_encoder(CFG)
decoder = get_decoder(CFG)
renderer = get_renderer(CFG)

CFG['training']['npoints_decoder'] = 10
#TODO view_cfg_path
dataset = get_synthetic_dataset(args.data, 'test', 'uniform', CFG, CAMS)

device = torch.device("cuda")
encoder = encoder.to(device)
decoder = decoder.to(device)
renderer = renderer.to(device)
encoder.eval()
decoder.eval()
renderer.eval()

#load params
checkpoint = torch.load(exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint), map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
if not CFG['renderer']['type'] == 'lfn':
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
renderer.load_state_dict(checkpoint['decoder_impl_state_dict'])

#iterate through dataset and store images
loader = dataset.get_loader()

rec_dir = exp_dir + 'recs/'

os.makedirs(rec_dir, exist_ok=True)
num_recs = 0
ious = []
mses = []
with torch.no_grad():
    for i, data in enumerate(loader):
        print('Rendering example {}'.format(i))
        num_recs += 1
        frame = data.get('frame').item()
        inp_pos = data.get('input_pos').to(device)
        inp_feats = data.get('input_feats').to(device)
        camera_params_tmp = data.get('camera_params')
        mask = [m.to(device).squeeze() for m in data.get('mask')]
        gt_depth = [d.to(device).squeeze() for d in data.get('depth_maps')]
        gt_color = [d.to(device).squeeze() for d in data.get('color_maps')]
        camera_params = [{k: v.to(device) for (k,v) in zip(c_params.keys(), c_params.values())} for c_params in camera_params_tmp]


        for c_idx in CAMS:
            xres = gt_depth[c_idx].shape[1] * args.res_factor
            yres = gt_depth[c_idx].shape[0] * args.res_factor
            xx, yy = np.meshgrid(np.arange(xres), np.arange(yres))
            if not CFG['renderer']['type'] == 'lfn':
                xx = xx / xres
                yy = yy / yres
            img_coords = torch.from_numpy(np.stack([xx, yy], axis=-1)).float().reshape(-1, 2).unsqueeze(0).to(device)

            z = encoder(inp_pos, inp_feats)
            encoding = decoder(z)
            if 'anchors' in encoding:
                encoding['anchors'] *= 2*radius
            kps_2d_ = get2Dkps(camera_params[c_idx], inp_pos*2*radius, gt_depth[c_idx])

            coord_chunks = torch.split(img_coords, args.npixels_per_batch, dim=1)
            logit_chunks = []
            depth_chunks = []
            color_chunks = []
            for coords in coord_chunks:
                chunk, chunk_d, chunk_c = renderer(coords, encoding, camera_params[c_idx], kps_2d_)
                logit_chunks.append(chunk.squeeze().detach())
                depth_chunks.append(chunk_d.squeeze().detach())
                color_chunks.append(chunk_c.squeeze().detach())

            logits = torch.cat(logit_chunks, dim=0)
            dephts = torch.cat(depth_chunks, dim=0)
            colors = torch.cat(color_chunks, dim=0)
            rec_img = torch.sigmoid(logits.reshape(yres, xres))
            rec_depth = dephts.reshape(yres, xres)
            red = colors[:, 0].reshape(yres, xres)
            green = colors[:, 1].reshape(yres, xres)
            blue = colors[:, 2].reshape(yres, xres)
            rec_color = torch.clamp(torch.stack([red, green, blue], dim=-1)*255, 0, 255)

            keyPos_hom = torch.cat([inp_pos*2*radius, torch.ones([inp_pos.shape[0], inp_pos.shape[1], 1], device=inp_pos.device, dtype=torch.float)], dim=2).permute(0, 2, 1)
            tmp = torch.bmm(camera_params[c_idx]['extrinsics'], keyPos_hom).permute(0, 2, 1)
            d = tmp[:, :, -1].squeeze()

            threshold = 0.5
            rec_depth[rec_img > threshold] = rec_depth[rec_img > threshold] * max(d-torch.min(d).item()) + torch.min(d).item()
            gt_depth[c_idx][mask[c_idx] > threshold] = gt_depth[c_idx][mask[c_idx] > threshold] * max(d-torch.min(d).item()) + torch.min(d).item()

            rec_depth[rec_img <= threshold] = -1.0
            gt_depth[c_idx][mask[c_idx] <= threshold] = -1.0
            rec_color[rec_img <= threshold] = 0

            np.save(rec_dir + '/rec_depth_frame{}_camera{}.npy'.format(frame, c_idx), rec_depth.detach().cpu().numpy())
            np.save(rec_dir + '/rec_color_frame{}_camera{}.npy'.format(frame, c_idx), rec_color.detach().cpu().numpy().astype(np.uint8))
            np.save(rec_dir + '/rec_mask_frame{}_camera{}.npy'.format(frame, c_idx), rec_img.detach().cpu().numpy())
            np.save(rec_dir + '/gt_depth_frame{}_camera{}.npy'.format(frame, c_idx), gt_depth[c_idx].detach().cpu().numpy())
            np.save(rec_dir + '/gt_color_frame{}_camera{}.npy'.format(frame, c_idx), (gt_color[c_idx].detach().cpu().numpy()*255).astype(np.uint8))
            np.save(rec_dir + '/gt_mask_frame{}_camera{}.npy'.format(frame, c_idx), (mask[c_idx].detach().cpu().numpy()))
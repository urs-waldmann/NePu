import torch
import numpy as np
import argparse, yaml
from torch.nn import functional as F
import os
from scipy.spatial.transform import Rotation as R

from data.synthDataset import get_synthetic_dataset
from models.NePu import get_encoder, get_decoder,  get_renderer
from models.utils import get2Dkps



upper_bound = {'pigeons': 0.2,
               'cows': 1.3,
               'humans': 1.0,
               'giraffes': 1.3}

iou_thresh = {
    'giraffes': 0.91,
    'pigeons': 0.95,
    'cows': 0.93,
    'humans': 0.92
}



parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-checkpoint', required=True, type=int)
parser.add_argument('-cams', nargs='+', required=True, type=int)
parser.add_argument('-data', required=True, type=str)
parser.add_argument('-npixels_per_batch', type=int, default=50000)

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


CFG['training']['npoints_decoder'] = 10000
dataset = get_synthetic_dataset(args.data, 'test', 'uniform', CFG, args.cams)
CAMS = dataset.cams


#TODO implement multi-view dataset for all classes
encoder = get_encoder(CFG).float()
decoder = get_decoder(CFG).float()
renderer = get_renderer(CFG).float()

encoder = encoder.to(device)
decoder = decoder.to(device)
renderer = renderer.to(device)
encoder.float()
decoder.float()
renderer.float()
encoder.eval()
decoder.eval()
renderer.eval()

checkpoint = torch.load(exp_dir + 'checkpoints/checkpoint_epoch_{}.tar'.format(args.checkpoint), map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
renderer.load_state_dict(checkpoint['decoder_impl_state_dict'])

cluster_centers = np.load('experiments/{}/cluster_centers_{}.npy'.format(args.exp_name, args.checkpoint))
print(cluster_centers.shape)

angles = [0, 90, 180, 270]

data = dataset.__getitem__(0)
inp_feats = data.get('input_feats').to(device).unsqueeze(0)

init_z = []
with torch.no_grad():
    for i in range(cluster_centers.shape[0]):
        z_ = torch.from_numpy(cluster_centers[i, :]).float().to(device).unsqueeze(0)
        encoding = decoder(z_)
        init_z.append(z_)
        for deg in angles[1:]:
            r = torch.from_numpy(R.from_euler('z', deg, degrees=True).as_matrix().T).to(device).float()
            kps_rot = encoding['anchors'] @  r
            init_z.append(encoder(kps_rot, inp_feats))

cam_name = 'all_cams'
if args.cams is not None:
    cam_name = ''
    for c in args.cams:
        cam_name += str(c)+'_'
    cam_name = cam_name[:-1]
exp_name = args.exp_name + '_{}_'.format(args.checkpoint) + cam_name + '_test'
os.makedirs('{}'.format(exp_name), exist_ok=True)


MPJMP = []
inds = []
succ_cluster = []
for ind in range(0, len(dataset)):
    print('NEW EXPERIEMNT!')
    print('Number: {}'.format(ind))
    if any([os.path.isdir('{}/sil{}_{}'.format(exp_name, ind, j)) for j in range(cluster_centers.shape[0])]):
        print('exists already, skip!')
        continue
    data_init = dataset.__getitem__(ind)
    inp_pos_init = data_init.get('input_pos').to(device).unsqueeze(0)

    data = dataset.__getitem__(ind)
    gt_pos = data.get('input_pos').to(device).unsqueeze(0)
    inp_feats = data.get('input_feats').to(device).unsqueeze(0)
    camera_params_tmp = data.get('camera_params')
    masks = data.get('mask')
    depth_maps_gt = data.get('depth_maps')
    sup_pos = [d.to(device).unsqueeze(0) for d in data.get('supervision_pos')]
    sup_gt = [d.to(device).unsqueeze(0) for d in data.get('supervision_gt')]
    camera_params = [{k: v.to(device).unsqueeze(0) for (k,v) in zip(c.keys(), c.values())} for c in camera_params_tmp]
    kps_2ds = data.get('kps_2ds')
    kps_2ds = [kps.to(device).unsqueeze(0) for kps in kps_2ds]

    radius = CFG['data']['radius']


    masks_shapes = torch.tensor([masks[idx].squeeze().shape for idx in range(len(CAMS))], device=device).squeeze()
    print(masks_shapes.shape)
    if len(masks_shapes.shape) < 2:
        masks_shapes = masks_shapes.unsqueeze(0)
    camera_params_stacked = {k: torch.cat([camera_params[idx][k] for idx in range(len(CAMS))], dim=0) for k in camera_params[0].keys()}
    sup_pos_stacked = torch.cat([sup_pos[idx] for idx in range(len(CAMS))], dim=0)

    for i, mask in enumerate(masks):
        mask[mask < 0.5] = 0.0
        mask[mask >= 0.5] = 1.0
        masks[i] = mask.to(device).squeeze()
    for i, depth_map in enumerate(depth_maps_gt):
        depth_maps_gt[i] = depth_map.to(device).squeeze()

    ls = []
    best_iou = 0
    kps_loss_final = np.NAN
    best_idx = None
    for ii in range(len(init_z)):

        if best_iou > iou_thresh[args.data]:
            break
        print('Cluster: {}, rot: {}'.format(int(ii/len(angles)), ii%len(angles)))
        z = init_z[ii].detach().clone()
        z.requires_grad = True

        ##set-up the optimizer
        opt = torch.optim.LBFGS([z], lr=1)


        def full_eval():
            encoding = decoder(z)
            kps_metric = encoding['anchors'] * 2 * radius
            kps_metric_stacked = kps_metric.repeat(len(CAMS), 1, 1)
            kps_2d_stacked = get2Dkps(camera_params_stacked,
                                      kps_metric_stacked, masks_shapes)
            encoding_stacked = {}
            encoding_stacked['anchors'] = kps_metric_stacked
            encoding_stacked['anchor_feats'] = encoding['anchor_feats'].repeat(len(CAMS), 1, 1)
            encoding_stacked['z'] = encoding['z'].repeat(len(CAMS), 1)
            all_logits, _, _ = renderer(sup_pos_stacked, encoding_stacked, camera_params_stacked, kps_2d_stacked)
            all_logits = all_logits.squeeze()
            all_gt = torch.cat([sup_gt[i] for i in range(len(CAMS))]).squeeze()
            rec_loss = F.binary_cross_entropy_with_logits(all_logits, all_gt, reduction='mean')
            reg_loss = torch.norm(z, dim=-1).mean()
            loss = rec_loss + reg_loss/10
            return encoding, loss, reg_loss, rec_loss

        def closure():
            opt.zero_grad()
            _, loss, _, _ = full_eval()
            loss.backward()
            return loss


        best_loss = 100.0
        best_state = None
        for i in range(10):
            opt.step(closure)

            with torch.no_grad():
                encoding, loss, reg_loss, rec_loss = full_eval()
                kps_loss = ((gt_pos*2*radius - encoding['anchors'].detach()*2*radius)**2).sum(dim=-1).sqrt().mean()

            print('Loss: {}, Kps loss: {}'.format(loss.item(), kps_loss.item()))
            if loss.item() > upper_bound[args.data]:
                break
            if loss.item() < best_loss:
                print('better')
                best_loss = loss.item()
                best_state = {'z': z.detach().clone(),
                              'kps': encoding['anchors'].detach().clone()*2*radius,
                              'feats': encoding['anchor_feats'].detach().clone()}
            else:
                print('worse')


        if best_state is not None:
            with torch.no_grad():
                xres = int(masks[0].shape[1])  # int(320)
                yres = int(masks[0].shape[0])  # int(240)
                xx, yy = np.meshgrid(np.arange(xres), np.arange(yres))
                xx = xx / xres
                yy = yy / yres
                img_coords = torch.from_numpy(np.stack([xx, yy], axis=-1)).float().reshape(-1, 2).unsqueeze(0).to(device)

                kps_2d_ = get2Dkps(camera_params[0], best_state['kps'], torch.tensor(masks[0].shape, device=device).unsqueeze(0))

                coord_chunks = torch.split(img_coords, args.npixels_per_batch, dim=1)
                logit_chunks = []
                depth_chunks = []
                color_chunks = []
                for coords in coord_chunks:
                    l_chunk, d_chunk, c_chunk = renderer(coords, {'z': best_state['z'],
                                                                      'anchors': best_state['kps'],
                                                                      'anchor_feats': best_state['feats']}, camera_params[0], kps_2d_)

                    logit_chunks.append(l_chunk.squeeze().detach())
                    depth_chunks.append(d_chunk.squeeze().detach())
                    color_chunks.append(c_chunk.squeeze().detach())

                logits = torch.cat(logit_chunks, dim=0)
                depth_map = torch.cat(depth_chunks, dim=0)
                color = torch.cat(color_chunks, dim=0)

                logits = logits.squeeze()
                depth_map = depth_map.squeeze()
                logits = logits.reshape(yres, xres)
                depth_map = depth_map.reshape(yres, xres)
                color = color.squeeze().reshape(yres, xres, 3)
                rec_img = logits.detach().cpu().numpy()
                rec_depth_map = depth_map.detach().cpu().numpy()
                rec_color = color.detach().cpu().numpy()
                os.mkdir('{}/sil{}_{}'.format(exp_name, ind, ii))
                np.save('{}/sil{}_{}/gt_kps.npy'.format(exp_name, ind, ii), gt_pos.detach().cpu().numpy()*2*radius)
                np.save('{}/sil{}_{}/gt_mask.npy'.format(exp_name, ind, ii), masks[0].detach().cpu().numpy())
                np.save('{}/sil{}_{}/gt_depth_map.npy'.format(exp_name, ind, ii), depth_maps_gt[0].detach().cpu().numpy())
                np.save('{}/sil{}_{}/rec.npy'.format(exp_name, ind, ii), rec_img)
                np.save('{}/sil{}_{}/rec_depth.npy'.format(exp_name, ind, ii), rec_depth_map)
                np.save('{}/sil{}_{}/rec_color.npy'.format(exp_name, ind, ii), rec_color)
                np.save('{}/sil{}_{}/rec_kps.npy'.format(exp_name, ind, ii), best_state['kps'].cpu().numpy())

                fore_gt = masks[0].detach().squeeze() > 0.5
                fore_pred = torch.sigmoid(logits.squeeze().detach()) > 0.5

                intersection = torch.logical_and(fore_gt, fore_pred).sum()
                union = torch.logical_or(fore_gt, fore_pred).sum()

                iou = (intersection/union).item()
                kps_loss = ((gt_pos*2*radius - best_state['kps'])**2).sum(dim=-1).sqrt().mean()
                print('IOU: {}, Loss: {}, Kps loss: {}'.format(iou, best_loss, kps_loss))

                if iou > best_iou:
                    best_iou = iou
                    kps_loss_final = kps_loss.item()
                    best_idx = ii

    if best_idx is None:
        print('ATTENTION: NO SOLUTION FOUND!!!')


    MPJMP.append(kps_loss_final)
    succ_cluster.append(best_idx)
    print('Current Average is {} after {} examples!'.format(np.nanmean(np.array(MPJMP)), len(MPJMP)))
    np.save('invOpt_{}_{}_inter_res_supp8cams.npy'.format(args.exp_name, args.checkpoint), np.array(MPJMP))
    np.save('invOpt_{}_{}_succ_clus_supp8cams.npy'.format(args.exp_name, args.checkpoint), np.array(succ_cluster))
    inds.append(ind)
    np.save('invOpt_{}_{}_example_inds_supp8cams.npy'.format(args.exp_name, args.checkpoint), np.array(inds))

print(MPJMP)
print(np.nanmean(np.array(MPJMP)))
print(succ_cluster)
print(inds)
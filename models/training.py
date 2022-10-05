from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import math
from time import time

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer(object):

    def __init__(self, encoder, decoder, decoder_impl, cfg, device, train_dataset, val_dataset, exp_name, optimizer='Adam'):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.decoder_impl = decoder_impl.to(device)
        print('Number of Parameters in encoder: {}'.format(count_parameters(self.encoder)))
        print('Number of Parameters in decoder: {}'.format(count_parameters(self.decoder)))
        self.cfg = cfg
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.AdamW(params=list(encoder.parameters()) +
                                                list(decoder.parameters()) +
                                                list(decoder_impl.parameters()),
                                         lr=self.cfg['lr'],
                                         weight_decay=0.001)
        self.lr = self.cfg['lr']

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.val_data_loader = self.val_dataset.get_loader()

    def reduce_lr(self, epoch):
        if epoch > 0 and self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * self.cfg['lr_decay_factor']**decay_steps

    def train_step(self, batch):
        #t0_idk = time()
        self.encoder.train()
        self.decoder.train()
        self.decoder_impl.train()
        self.optimizer.zero_grad()
        loss_mask, loss_kps, loss_depth, loss_color, loss_reg = self.compute_loss(batch)

        loss_tot = loss_mask + loss_kps + loss_reg/32 + loss_depth + loss_color
        loss_tot.backward()
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.decoder_impl.parameters(), max_norm=self.cfg['grad_clip'])
        self.optimizer.step()
        return loss_tot.item(), loss_mask.item(), loss_kps.item(), loss_depth.item(), loss_color.item(), loss_reg.item()

    def compute_loss(self, batch):
        device = self.device

        sup_pos = batch.get('supervision_pos')[0].to(device)
        sup_gt = batch.get('supervision_gt')[0].to(device).squeeze()
        inp_pos_pert = batch.get('input_pos_pert').to(device)

        inp_pos = batch.get('input_pos').to(device)
        radius = batch.get('radius').to(device)
        inp_feats = batch.get('input_feats').to(device)
        depth_pos = batch.get('supervision_pos_fore')[0].to(device)
        depth = batch.get('supervision_gt_depth')[0].to(device)
        color = batch.get('supervision_gt_color')[0].to(device)
        kps_2d = batch.get('kps_2ds')[0].to(device)

        camera_params_tmp = batch.get('camera_params')[0]
        camera_params = {k: v.to(device) for (k,v) in zip(camera_params_tmp.keys(), camera_params_tmp.values())}

        z = self.encoder(inp_pos_pert, inp_feats)
        encoding = self.decoder(z)
        if 'anchors' in encoding:
            loss_kps = torch.sum((inp_pos - encoding['anchors'])**2, dim=-1).sqrt().mean()
            encoding['anchors'] = encoding['anchors']*2*radius.unsqueeze(1).unsqueeze(2)
        else:
            loss_kps = torch.zeros_like(inp_pos).mean()
        logits, _, _ = self.decoder_impl(sup_pos, encoding, camera_params, kps_2d)
        logits = logits.squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, sup_gt.squeeze(), reduction='mean')

        _, depth_pred, color_pred = self.decoder_impl(depth_pos, encoding, camera_params, kps_2d)
        depth_pred = depth_pred.squeeze()
        color_pred = color_pred.squeeze()
        loss_depth = ((depth - depth_pred)**2).mean()
        loss_color = ((color - color_pred)**2).mean()

        return loss, loss_kps, loss_depth, loss_color, torch.norm(z, dim=-1).mean()


    def train_model(self, epochs, ckp_interval=5):
        loss = 0
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss = 0
            sum_loss_mask = 0
            sum_loss_kps = 0
            sum_loss_depth = 0
            sum_loss_color = 0
            sum_loss_reg = 0
            train_data_loader = self.train_dataset.get_loader()

            for batch in train_data_loader:
                loss, loss_mask, loss_kps, loss_depth, loss_color, loss_reg = self.train_step(batch)

                sum_loss += loss
                sum_loss_mask += loss_mask
                sum_loss_kps += loss_kps
                sum_loss_depth += loss_depth
                sum_loss_color += loss_color
                sum_loss_reg += loss_reg

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)
            self.writer.add_scalar('training loss mask batch avg', sum_loss_mask / len(train_data_loader), epoch)
            self.writer.add_scalar('training loss kps batch avg', sum_loss_kps / len(train_data_loader), epoch)
            self.writer.add_scalar('training loss depth batch avg', sum_loss_depth / len(train_data_loader), epoch)
            self.writer.add_scalar('training loss color batch avg', sum_loss_color / len(train_data_loader), epoch)
            self.writer.add_scalar('training loss reg batch avg', sum_loss_reg / len(train_data_loader), epoch)



            if epoch % ckp_interval == 0:
                self.save_checkpoint(epoch)
                val_loss, val_loss_mask, val_loss_kps, val_loss_depth, val_loss_color, val_loss_reg = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])

                self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                self.writer.add_scalar('val loss mask batch avg', val_loss_mask, epoch)
                self.writer.add_scalar('val loss kps batch avg', val_loss_kps, epoch)
                self.writer.add_scalar('val loss depth batch avg', val_loss_depth, epoch)
                self.writer.add_scalar('val loss color batch avg', val_loss_color, epoch)
                self.writer.add_scalar('val loss reg batch  avg', val_loss_reg, epoch)
                print("Epoch {:5d}: KPSloss: {:06.4f} - {:06.4f}, DepthLoss: {:06.4f} - {:06.4f}, "
                      "ColorLoss: {:06.4f} - {:06.4f}, MaskLoss: {:06.4f} - {:06.4f}, Total Loss: {:06.4f} - {:06.4f}".format(
                    epoch,
                    sum_loss_kps/len(train_data_loader),
                    val_loss_kps,
                    sum_loss_depth/len(train_data_loader),
                    val_loss_depth,
                    sum_loss_color/len(train_data_loader),
                    val_loss_color,
                    sum_loss_mask/len(train_data_loader),
                    val_loss_mask,
                    sum_loss/len(train_data_loader),
                    val_loss))

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):

            torch.save({'epoch': epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'decoder_impl_state_dict': self.decoder_impl.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        if self.cfg['ckpt'] is not None:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(self.cfg['ckpt'])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.decoder_impl.load_state_dict(checkpoint['decoder_impl_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        for param_group in self.optimizer.param_groups:
            print('Setting LR to {}'.format(self.cfg['lr']))
            param_group['lr'] = self.cfg['lr']
        if self.cfg['lr_decay_interval'] is not None:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * self.cfg['lr_decay_factor']**decay_steps
        return epoch

    def compute_val_loss(self):
        self.encoder.eval()
        self.decoder.eval()
        self.decoder_impl.eval()

        sum_val_loss = 0
        sum_val_mask = 0
        sum_val_kps = 0
        sum_val_depth = 0
        sum_val_color = 0
        sum_val_reg = 0


        c = 0
        for val_batch in self.val_data_loader:
            l_mask, l_kps, l_depth, l_color, l_reg = self.compute_loss(val_batch)
            sum_val_mask += l_mask.item()
            sum_val_kps += l_kps.item()
            sum_val_depth += l_depth.item()
            sum_val_color += l_color.item()
            sum_val_reg += l_reg.item()
            sum_val_loss += l_mask.item() + l_kps.item() + l_depth.item() + l_color.item() + l_reg.item()/32
            c = c + 1



        return sum_val_loss / c,  sum_val_mask / c, sum_val_kps / c, sum_val_depth / c, sum_val_color / c, sum_val_reg / c

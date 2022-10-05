import math
from math import floor
import numpy as np
import numpy.random
from torch.utils.data import Dataset
import torch
import os
import configparser
import cv2
import imageio
from scipy.signal import convolve2d
from scipy.spatial.transform import Rotation as R

from .utils import get_intrinsic_params, get_extrinsic_params, proj



class SilhouetteDatasetReal(Dataset):
    def __init__(self, type, mode, n_supervision_points, batch_size, nkps, num_datapoints, radius, ncams, cams=None):

        self.type = type
        self.mode = mode
        self.nkps = nkps
        self.ncams = ncams
        self.N = num_datapoints
        self.radius = radius

        np.random.seed(0)

        paths = {'pigeons': '/abyss/home/silhouettes/',
                 'humans': '/abyss/home/silhouettes/',
                 'cows': '/abyss/home/silhouettes/',
                 'giraffes': '/abyss/home/silhouettes/'
                 }
        self.path = paths[type]
        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points

        paths_kps = {'pigeons': '/abyss/home/silhouettes/synth_data/',
                     'humans': '/abyss/home/silhouettes/multiview_smplx/male/pose_512/',
                     'cows': '/abyss/home/silhouettes/multiview_cow/cow_animation_Holstein.blend/',
                     'giraffes': '/abyss/home/silhouettes/multiview_giraffe/'
                     }
        self.keyPos3d = np.genfromtxt(paths_kps[type] + 'keypoints.csv', delimiter=',')[1:, 1:]
        if type == 'pigeons':
            self.keyPos3d = np.concatenate([self.keyPos3d[:, :7*3], self.keyPos3d[:, 9*3:]], axis=1)

        self.keyPos3d = self.keyPos3d.reshape((self.keyPos3d.shape[0], -1, 3))
        print(self.keyPos3d.shape)

        if cams is None:
            self.cams = range(self.ncams)
        else:
            self.cams = cams


        # the maximum and minimum values per spatial dimension are:
        # [0.1713163  0.16732147 0.1758557 ]
        # [-0.1597985  -0.15980852 -0.01288748]
        #TODO: provide normalized and unnormalized for projection
        #TODO: or find global normalization
        # self.keyPos3d = self.keyPos3d * np.expand_dims(np.array([5, 5, 5]), axis=[0, 1])

        parser = configparser.ConfigParser()

        cfgs = parser.read(paths_kps[type] + 'parameters_test.cfg')

        self.cameraIntrinsics = get_intrinsic_params(parser)

        self.cameraExtrinsics = []
        self.cameraExtrinsics_all = []

        self.trans = []


        for i in self.cams:
            print('LOADING CAM: {}'.format(i))
            RT, R, t = get_extrinsic_params(parser, cam_id=i)
            self.cameraExtrinsics.append(RT)
            self.trans.append(t)

        for i in range(24):
            print('LOADING CAM: {}'.format(i))
            RT, R, t = get_extrinsic_params(parser, cam_id=i)
            self.cameraExtrinsics_all.append(RT)



    def __len__(self):
        return 4

    def sample_points2(self, n_samps, mask, keyPos3d, camera_params):
        center = np.mean(keyPos3d, axis=0)
        radius = self.radius
        offsets = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    input = [0, 0, 0]
                    input[0] = x * radius
                    input[1] = y * radius
                    input[2] = z * radius
                    offsets.append(np.array(input))

        off = np.stack(offsets, axis=0)
        corners = np.expand_dims(center, axis=0) + off

        corners_2d = proj(camera_params, corners)
        lower_b = np.clip(np.min(corners_2d, axis=0), 0, None)
        upper_b = np.max(corners_2d, axis=0)
        upper_b[0] = min(upper_b[0], mask.shape[1]-1)+1
        upper_b[1] = min(upper_b[1], mask.shape[0]-1)+1
        fore = np.where(mask == 0)
        back = np.where(mask == 1)
        fore_x = fore[0]
        fore_y = fore[1]
        back_x = back[0]
        back_y = back[1]
        fore_x_valid = np.logical_and(fore_x > lower_b[0], fore_x < upper_b[0])
        fore_y_valid = np.logical_and(fore_y > lower_b[1], fore_y < upper_b[1])
        back_x_valid = np.logical_and(back_x > lower_b[0], back_x < upper_b[0])
        back_y_valid = np.logical_and(back_y > lower_b[1], back_y < upper_b[1])
        fore_x = fore_x[np.logical_and(fore_x_valid, fore_y_valid)]
        fore_y = fore_y[np.logical_and(fore_x_valid, fore_y_valid)]
        back_x = back_x[np.logical_and(back_x_valid, back_y_valid)]
        back_y = back_y[np.logical_and(back_x_valid, back_y_valid)]
        res = mask.shape
        samps_fore = np.random.randint(0, len(fore_x), int(n_samps/2))
        samps_x_fore = fore_x[samps_fore]
        samps_y_fore = fore_y[samps_fore]
        samps_back = np.random.randint(0, len(back_x), n_samps-int(n_samps/2))
        samps_x_back = back_x[samps_back]
        samps_y_back = back_y[samps_back]
        # flip axes to meet properties of 2D keypoints
        pos_fore = np.stack([samps_y_fore/res[1], samps_x_fore/res[0]], axis=1)
        pos_back = np.stack([samps_y_back/res[1], samps_x_back/res[0]], axis=1)

        gt_fore = mask[samps_x_fore, samps_y_fore]
        gt_back = mask[samps_x_back, samps_y_back]

        return np.concatenate([pos_fore, pos_back], axis=0), np.concatenate([gt_fore, gt_back], axis=0)


    def sample_points(self, n_samps, idx, mask):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]
        # flip axes to meet properties of 2D keypoints
        return mask[samps_x, samps_y], np.stack([samps_y/res[1], samps_x/res[0]], axis=1)


    def __getitem__(self, idx):

        idx += 1
        #images = ['real_giraffe_standing.png',
        #          'real_giraffe_drinking.png']
        #images = ['cow0.png']
        #images = ['pigeon0.png']
        images = ['human1.png']

        if self.type == 'pigeons':
            keyPos3d = self.keyPos3d[idx, :, :]
        else:
            keyPos3d = self.keyPos3d[14, :, :] #1158, :, :]#14
        onehot = np.eye(self.nkps)
        masks = []
        depth_maps = []
        color_maps = []
        sup_pos = []
        sup_gt = []
        sup_col = []
        sup_dep = []
        sup_pos_foreground = []
        cams = []
        kps_2ds = []

        CAMS = [0]
        for j, camera_idx in enumerate(CAMS):
            path = self.path + images[idx-1]
            print(images[idx-1])

            intrinsics = self.cameraIntrinsics
            extrinsics = self.cameraExtrinsics[j]
            t_inv = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
            R_inv = extrinsics[:3, :3].T
            ex_inv = np.zeros((3, 4))
            ex_inv[:3, :3] = R_inv
            ex_inv[:3, 3] = t_inv
            trans = self.trans[j]
            if self.type == 'humans1024':
                path = self.path + '{:03d}_1024/'.format(camera_idx)
                mask = imageio.imread(path + 'objectID{:04d}.{:03d}.png'.format(idx, camera_idx))
                ####mask[mask<255] = 0
                ####mask = cv2.resize(mask, (1024, 1024))
                ####mask[mask > 255/2] = 255
                ####mask[mask <= 255/2] = 0
                ####imageio.imsave(self.path + '{:03d}_1024/objectID{:04d}.{:03d}.png'.format(camera_idx_ori, idx, camera_idx_ori), mask)

                #color = np.flip(self.color[camera_idx_ori, idx-1, :, :, :].astype(np.float32), axis=-1)
                color = imageio.imread(path + 'Image{:04d}.{:03d}.png'.format(idx, camera_idx))
                ####color = cv2.resize(color, (1024, 1024))
                ####imageio.imsave(self.path + '{:03d}_1024/Image{:04d}.{:03d}.png'.format(camera_idx_ori, idx, camera_idx_ori), color)
                color = color.astype(np.float32)
                color = color / 255

                depth = np.load(path + 'depth{:04d}.{:03d}.npy'.format(idx, camera_idx))
            else:
                if self.type == 'pigeons':
                    mask = imageio.imread(path)[:, :, 0]
                else:
                    mask = imageio.imread(path)[:, :, 0]
                    mask[mask<255] = 0
            # obtain standard sized crop around subject
            center = np.mean(keyPos3d, axis=0)
            radius = self.radius
            offsets = []
            for x in [-1, 1]:
                for y in [-1, 1]:
                    for z in [-1, 1]:
                        input = [0, 0, 0]
                        input[0] = x * radius
                        input[1] = y * radius
                        input[2] = z * radius
                        offsets.append(np.array(input))

            off = np.stack(offsets, axis=0)
            corners = np.expand_dims(center, axis=0) + off

            camera_params = {'intrinsics': intrinsics, 'extrinsics': extrinsics}
            kps_2d = proj(camera_params, keyPos3d)
            corners_2d = proj(camera_params, corners)
            lower_b = np.clip(np.min(corners_2d, axis=0), 0, None)
            upper_b = np.max(corners_2d, axis=0)
            upper_b[0] = min(upper_b[0], mask.shape[1]-1)+1
            upper_b[1] = min(upper_b[1], mask.shape[0]-1)+1
            #restrict depth and mask!
            CROP = False
            if CROP:
                mask = mask[math.ceil(lower_b[1]):math.floor(upper_b[1]), math.ceil(lower_b[0]):math.floor(upper_b[0])]
                kps_2d = kps_2d - lower_b
            kps_2d[:, 0] /= mask.shape[1]
            kps_2d[:, 1] /= mask.shape[0]


            mask[mask==255] = 1
            n_far = int(self.n_supervision_points/2)
            n_near = self.n_supervision_points - n_far
            #sup_pos_, sup_gt_ = self.sample_points2(self.n_supervision_points, mask, keyPos3d, camera_params)
            ####sup_gt_far, sup_pos_far = self.sample_points(n_far, interesting_pos, mask, 0.05)#0.15)
            ####sup_gt_near, sup_pos_near = self.sample_points(n_near, interesting_pos, mask, 0.05) #0.015)
            background_idx = np.where(mask == 0)
            foreground_idx = np.where(mask == 1)

            sup_gt_fore, sup_pos_fore = self.sample_points(n_far, foreground_idx, mask)
            sup_gt_back,  sup_pos_back = self.sample_points(n_near, background_idx, mask)
            sup_pos_ = np.concatenate([sup_pos_fore, sup_pos_back], axis=0)
            sup_gt_ = np.concatenate([sup_gt_fore, sup_gt_back], axis=0)

            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float(),
                         'ex_inv': torch.from_numpy(ex_inv).float(),
                         'trans': torch.from_numpy(trans).float()})
            masks.append(torch.from_numpy(mask))
            sup_pos.append(torch.from_numpy(sup_pos_).float())
            sup_gt.append(torch.from_numpy(sup_gt_).float())
            sup_pos_foreground.append(torch.from_numpy(sup_pos_fore).float())
            kps_2ds.append(torch.from_numpy(kps_2d).float())


        return {'input_pos': torch.from_numpy(keyPos3d).float() / (2*radius),
                'input_feats': torch.from_numpy(onehot).float(),
                'supervision_pos': sup_pos,
                'supervision_gt': sup_gt,
                'supervision_pos_fore': sup_pos_foreground,
                'camera_params': cams,
                'mask': masks,
                'kps_2ds': kps_2ds,
                'frame': idx,
                'extrinsics_all': [torch.from_numpy(params).float() for params in self.cameraExtrinsics_all]}


    def get_loader(self, shuffle=True):
        #random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=10, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)



class SyntheticDataset(Dataset):
    def __init__(self, type, mode, n_supervision_points, supervision_distr, batch_size,
                 nkps, num_datapoints, radius, ncams,
                 cams=None, normalize_pixel_coords=True, noise_aug=None, cam_aug=True):

        self.type = type
        self.mode = mode
        self.nkps = nkps
        self.ncams = ncams

        self.noise_aug = noise_aug
        self.cam_aug = cam_aug

        N = num_datapoints
        self.N = N

        self.radius = radius
        np.random.seed(0)

        # compute splits
        if type == 'humans':
            perm = np.random.permutation(N)
            time_steps = np.arange(N) + 1
            perm_steps = time_steps[perm]
            n_train = int(N*0.70)
            n_val = int(N*0.1)
            n_test = N - n_train - n_val
            train_steps = perm_steps[:n_train]
            val_steps = perm_steps[n_train:n_train+n_val]
            test_steps = perm_steps[-n_test:]
            print(len(test_steps))
        else:
            n_splits = int(N/10)
            n_train = floor(n_splits*0.7)
            n_val = floor(n_splits*0.1)
            n_test = n_splits - n_train - n_val

            A = np.arange(N)
            # TODO: update once dataset is updated!
            if type != 'pigeons':
                A += 1

            chunks = np.stack(np.split(A, n_splits), axis=0)

            perm = np.random.permutation(n_splits)

            chunks = chunks[perm, :]

            chunks_train = chunks[:n_train, :]
            chunks_val = chunks[n_train:n_train+n_val, :]
            chunks_test = chunks[-n_test:, :]


            train_steps = np.reshape(chunks_train, [-1])
            val_steps = np.reshape(chunks_val, [-1])
            test_steps = np.reshape(chunks_test, [-1])

        steps = {'train': train_steps, 'val': val_steps, 'test': test_steps}
        self.steps = steps[mode]

        paths = {'pigeons': '/abyss/home/silhouettes/synth_data/',
                 'humans': '/abyss/home/silhouettes/multiview_smplx/male/pose_512/',
                 'cows': '/abyss/home/silhouettes/multiview_cow/cow_animation_Holstein.blend/',
                 'giraffes': '/abyss/home/silhouettes/multiview_giraffe/'
                 }
        self.path = paths[type]
        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points
        self.supervision_distr = supervision_distr
        self.NORM = normalize_pixel_coords

        self.keyPos3d = np.genfromtxt(self.path + 'keypoints.csv', delimiter=',')[1:, 1:]
        #TODO n_cams is different for novel views!!!
        self.n_cams = 24
        # for pigeons remove the unfolded wing keypoints and use 56 cameras
        if type == 'pigeons':
            self.keyPos3d = np.concatenate([self.keyPos3d[:, :7*3], self.keyPos3d[:, 9*3:]], axis=1)
            self.n_cams = 56

        self.keyPos3d = self.keyPos3d.reshape((self.keyPos3d.shape[0], -1, 3))
        print(self.keyPos3d.shape)

        self.cams = cams

        # obtain intrinsic and extrinsic parameters
        parser = configparser.ConfigParser()
        cfgs = parser.read(self.path + 'parameters.cfg')

        self.cameraIntrinsics = get_intrinsic_params(parser)

        self.cameraExtrinsics = []
        self.trans = []

        for i in range(self.n_cams):
            RT, R, t = get_extrinsic_params(parser, cam_id=i)
            self.cameraExtrinsics.append(RT)
            self.trans.append(t)

    def __len__(self):
        return len(self.steps)

    def sample_points_concentrated(self, n_samps, idx, mask, std):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]

        samps_y01 = samps_y/res[1] + np.random.randn(samps_y.shape[0])*std
        samps_x01 = samps_x/res[0] + np.random.randn(samps_x.shape[0])*std
        samps_x = np.clip((samps_x01*res[0]).astype(np.int32), 0, res[0]-1)
        samps_y = np.clip((samps_y01*res[1]).astype(np.int32), 0, res[1]-1)

        if not self.NORM:
            samps_y01 = samps_y
            samps_x01 = samps_x

        return mask[samps_x, samps_y], np.stack([samps_y01, samps_x01], axis=1)

    def sample_points_uniform(self, n_samps, idx, mask, depth, color):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]

        if self.NORM:
            samps_pos = np.stack([samps_y/res[1], samps_x/res[0]], axis=1)
        else:
            samps_pos = np.stack([samps_y, samps_x], axis=1)

        return mask[samps_x, samps_y], \
               depth[samps_x, samps_y], \
               color[samps_x, samps_y, :], \
               samps_pos

    def __getitem__(self, idx):
        idx = self.steps[idx]

        # TODO: update once dataset is updated
        if self.type == 'pigeons':
            keyPos3d = self.keyPos3d[idx, :, :]
        else:
            keyPos3d = self.keyPos3d[idx-1, :, :]
        onehot = np.eye(self.nkps)
        masks = []
        depth_maps = []
        color_maps = []
        sup_pos = []
        sup_gt = []
        sup_col = []
        sup_dep = []
        sup_pos_foreground = []
        cams = []
        kps_2ds = []

        cams_curr = self.cams
        if cams_curr is None:
            cams_curr = [np.random.randint(0, self.n_cams)]

        # when training we augment by a cyclic shift of the cameras and accordingly rotating the 3D keypoints
        # This helps the model to learn rotation equivariance
        if self.supervision_distr == 'training' and self.cam_aug and not self.mode == 'test':
            rot_steps = np.random.randint(0, 8)
        else:
            rot_steps = 0
            print('No rot aug!')
        # rotated 3D keypoints into view of "virtual camera"
        r = R.from_euler('z', 45*rot_steps, degrees=True).as_matrix()
        keyPos3d_unperturbed = keyPos3d
        if self.noise_aug is not None:
            keyPos3d = keyPos3d + np.random.randn(*list(keyPos3d.shape)) * self.noise_aug
        keyPos3d_unperturbed = (r @ keyPos3d_unperturbed.T).T
        keyPos3d = (r @ keyPos3d.T).T


        for camera_idx in cams_curr:
            # calculate index of "virtual camera"
            ring = math.floor(camera_idx/8)
            camera_idx_rot = ((camera_idx % 8) + rot_steps) % 8 + ring*8

            path = self.path + '{:03d}/'.format(camera_idx)

            intrinsics = self.cameraIntrinsics
            extrinsics = self.cameraExtrinsics[camera_idx_rot]
            t_inv = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
            R_inv = extrinsics[:3, :3].T
            ex_inv = np.zeros((3, 4))
            ex_inv[:3, :3] = R_inv
            ex_inv[:3, 3] = t_inv
            trans = self.trans[camera_idx_rot]

            if self.type == 'pigeons':
                color = imageio.imread(path + 'Image{:04d}.{:03d}.png'.format(idx, camera_idx)).astype(np.float32)
                mask = imageio.imread(path + 'objectID{:04d}.{:03d}.png'.format(idx, camera_idx))[:, :, 0]
            else:
                color = imageio.imread(path + 'Image{:04d}.{:03d}.jpg'.format(idx, camera_idx)).astype(np.float32)
                mask = imageio.imread(path + 'objectID{:04d}.{:03d}.png'.format(idx, camera_idx))
                mask[mask<255] = 0
            color = color / 255
            depth = cv2.imread(path + 'depth{:04d}.{:03d}.exr'.format(idx, camera_idx),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth = depth[:, :, 0]


            camera_params = {'intrinsics': intrinsics, 'extrinsics': extrinsics}
            kps_2d = proj(camera_params, keyPos3d)

            if self.NORM:
                kps_2d[:, 0] /= mask.shape[1]
                kps_2d[:, 1] /= mask.shape[0]

            # normalize depth map, such that depth of closest key point is 0 and depth of farthest keypoint is 1
            keyPos_hom = np.transpose(np.concatenate([keyPos3d, np.ones([keyPos3d.shape[0], 1])], axis=1))
            tmp = np.transpose(extrinsics @ keyPos_hom)
            d = tmp[:, -1]
            min_d = np.min(d, axis=0)
            depth -= min_d
            depth = depth / np.max(d-min_d, axis=0)
            depth[mask != 255] = -1.0

            mask[mask==255] = 1
            n_far = int(self.n_supervision_points/2)
            n_near = self.n_supervision_points - n_far
            # do uniform sampling for silhouette supervision if
            # a) it is specified, which is the case for inverse rendering
            # b) LFN baseline during training (for this baseline concentrated is too complex)
            if self.supervision_distr == 'uniform' or (self.supervision_distr == 'training' and not self.NORM):
                background_idx = np.where(mask == 0)
                foreground_idx = np.where(mask == 1)
                sup_gt_fore, depth_gt, color_gt, sup_pos_fore = self.sample_points_uniform(n_far, foreground_idx, mask, depth, color)
                sup_gt_back, _, _, sup_pos_back = self.sample_points_uniform(n_near, background_idx, mask, depth, color)
                sup_pos_ = np.concatenate([sup_pos_fore, sup_pos_back], axis=0)
                sup_gt_ = np.concatenate([sup_gt_fore, sup_gt_back], axis=0)
            elif self.supervision_distr == 'training':
                # obtain boundary pixels
                filtered = convolve2d(mask.astype(np.int32), np.ones([3, 3], dtype=np.int32), mode='same')
                interesting_bool = np.logical_not(np.logical_or(filtered == 0, filtered == 255*9))
                interesting_pos = np.where(interesting_bool)
                sup_gt_far, sup_pos_far = self.sample_points_concentrated(n_far, interesting_pos, mask, 0.15)#0.075)
                sup_gt_near, sup_pos_near = self.sample_points_concentrated(n_near, interesting_pos, mask, 0.015)#0.0075)
                sup_pos_ = np.concatenate([sup_pos_near, sup_pos_far], axis=0)
                sup_gt_ = np.concatenate([sup_gt_near, sup_gt_far], axis=0)
                #TODO add parameters for n_sup_points for color and depth!
                _, depth_gt, color_gt, sup_pos_fore = self.sample_points_uniform(500, np.where(mask == 1), mask, depth, color)



            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float(),
                         'ex_inv': torch.from_numpy(ex_inv).float(),
                         'trans': torch.from_numpy(trans).float()})
            masks.append(torch.from_numpy(mask))
            sup_pos.append(torch.from_numpy(sup_pos_).float())
            sup_gt.append(torch.from_numpy(sup_gt_).float())
            sup_col.append(torch.from_numpy(color_gt).float())
            sup_dep.append(torch.from_numpy(depth_gt).float())
            sup_pos_foreground.append(torch.from_numpy(sup_pos_fore).float())
            kps_2ds.append(torch.from_numpy(kps_2d).float())
            depth_maps.append(torch.from_numpy(depth).float())
            color_maps.append(torch.from_numpy(color).float())


        r = R.from_euler('z', np.random.uniform(0, 360), degrees=True).as_matrix()
        keyPos3d_rot = (r @ keyPos3d.T).T
        keyPos3d_rot_unperturbed = (r @ keyPos3d_unperturbed.T).T

        return {'input_pos': torch.from_numpy(keyPos3d_unperturbed).float() / (2*self.radius),
                'input_pos_pert': torch.from_numpy(keyPos3d).float() / (2*self.radius),
                'radius': torch.tensor(self.radius).float(),
                'input_feats': torch.from_numpy(onehot).float(),
                'supervision_pos': sup_pos,
                'supervision_gt': sup_gt,
                'supervision_gt_depth': sup_dep,
                'supervision_gt_color': sup_col,
                'supervision_pos_fore': sup_pos_foreground,
                'camera_params': cams,
                'mask': masks,
                'kps_2ds': kps_2ds,
                'depth_maps': depth_maps,
                'color_maps': color_maps,
                'frame': idx,
                'input_pos_rot': torch.from_numpy(keyPos3d_rot_unperturbed).float() / (2*self.radius),
                'input_pos_pert_rot': torch.from_numpy(keyPos3d_rot).float() / (2*self.radius)}

    def get_loader(self, shuffle=True):
        #random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=10, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


class SilhouetteDatasetNovelViews(Dataset):
    def __init__(self, type, mode, n_supervision_points, batch_size, nkps, num_datapoints, radius, cams=None):

        self.type = type
        self.mode = mode
        self.nkps = nkps

        np.random.seed(0)
        N = num_datapoints
        self.N = N
        self.radius = radius

        if type == 'humans':
            perm = np.random.permutation(N)
            time_steps = np.arange(N) + 1
            perm_steps = time_steps[perm]
            n_train = int(N*0.70)
            n_val = int(N*0.1)
            n_test = N - n_train - n_val
            train_steps = perm_steps[:n_train]
            val_steps = perm_steps[n_train:n_train+n_val]
            test_steps = perm_steps[-n_test:]
            print(len(test_steps))
        else:
            n_splits = int(N/10)
            n_train = floor(n_splits*0.7)
            n_val = floor(n_splits*0.1)
            n_test = n_splits - n_train - n_val

            A = np.arange(N)
            if type != 'pigeons':
                A += 1

            chunks = np.stack(np.split(A, n_splits), axis=0)

            perm = np.random.permutation(n_splits)

            chunks = chunks[perm, :]

            chunks_train = chunks[:n_train, :]
            chunks_val = chunks[n_train:n_train+n_val, :]
            chunks_test = chunks[-n_test:, :]


            train_steps = np.reshape(chunks_train, [-1])
            val_steps = np.reshape(chunks_val, [-1])
            test_steps = np.reshape(chunks_test, [-1])


        steps = {'train': train_steps, 'val': val_steps, 'test': test_steps}

        self.steps = steps[mode]
        print(len(self.steps))
        paths = {'pigeons': '/abyss/home/silhouettes/synth_data/',
                 'humans': '/abyss/home/silhouettes/multiview_smplx/male/pose_512/',
                 'cows': '/abyss/home/silhouettes/multiview_cow/cow_animation_Holstein.blend/',
                 'giraffes': '/abyss/home/silhouettes/multiview_giraffe/'
                 }
        self.path = paths[type]
        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points

        self.keyPos3d = np.genfromtxt(self.path + 'keypoints.csv', delimiter=',')[1:, 1:]
        n_cams = 360
        if type == 'pigeons':
            self.keyPos3d = np.concatenate([self.keyPos3d[:, :7*3], self.keyPos3d[:, 9*3:]], axis=1)

        self.keyPos3d = self.keyPos3d.reshape((self.keyPos3d.shape[0], -1, 3))
        print(self.keyPos3d.shape)

        if cams is None:
            self.cams = range(n_cams)
        else:
            self.cams = cams


        # the maximum and minimum values per spatial dimension are:
        # [0.1713163  0.16732147 0.1758557 ]
        # [-0.1597985  -0.15980852 -0.01288748]
        #TODO: provide normalized and unnormalized for projection
        #TODO: or find global normalization
        # self.keyPos3d = self.keyPos3d * np.expand_dims(np.array([5, 5, 5]), axis=[0, 1])

        parser = configparser.ConfigParser()

        #cfgs = parser.read('/abyss/home/silhouettes/parameters_0.cfg')
        cfgs = parser.read('/abyss/home/silhouettes/parameters_diagonal.cfg')


        self.cameraIntrinsics = get_intrinsic_params(parser)
        self.cameraExtrinsics = []
        self.trans = []


        #scaling:
        # pigeon: 0.154
        # giraffe: 1.85
        # cow: 0.885
        # human: 0.739

        scale = {'pigeons': 0.154,
                 'giraffes': 1.85,
                 'cows': 0.885,
                 'humans': 0.739}
        offset = {'cows': 0.83,
                  'pigeons': 0,
                  'giraffes': 2.519,
                  'humans': 1.074}

        for i in self.cams:
            RT, R, t = get_extrinsic_params(parser, cam_id=i, scale_distance=scale[type], z_offset=offset[type])#1.8) #0.154
            #RT, R, t = get_extrinsic_params(parser, cam_id=i+1, scale_distance=scale[type]*10/4.5, z_offset=offset[type])#1.8) #0.154

            self.cameraExtrinsics.append(RT)
            self.trans.append(t)



    def __len__(self):
        return len(self.steps)

    def sample_points2(self, n_samps, mask, keyPos3d, camera_params):
        center = np.mean(keyPos3d, axis=0)
        radius = self.radius
        offsets = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    input = [0, 0, 0]
                    input[0] = x * radius
                    input[1] = y * radius
                    input[2] = z * radius
                    offsets.append(np.array(input))

        off = np.stack(offsets, axis=0)
        corners = np.expand_dims(center, axis=0) + off

        corners_2d = proj(camera_params, corners)
        lower_b = np.clip(np.min(corners_2d, axis=0), 0, None)
        upper_b = np.max(corners_2d, axis=0)
        upper_b[0] = min(upper_b[0], mask.shape[1]-1)+1
        upper_b[1] = min(upper_b[1], mask.shape[0]-1)+1
        fore = np.where(mask == 0)
        back = np.where(mask == 1)
        fore_x = fore[0]
        fore_y = fore[1]
        back_x = back[0]
        back_y = back[1]
        fore_x_valid = np.logical_and(fore_x > lower_b[0], fore_x < upper_b[0])
        fore_y_valid = np.logical_and(fore_y > lower_b[1], fore_y < upper_b[1])
        back_x_valid = np.logical_and(back_x > lower_b[0], back_x < upper_b[0])
        back_y_valid = np.logical_and(back_y > lower_b[1], back_y < upper_b[1])
        fore_x = fore_x[np.logical_and(fore_x_valid, fore_y_valid)]
        fore_y = fore_y[np.logical_and(fore_x_valid, fore_y_valid)]
        back_x = back_x[np.logical_and(back_x_valid, back_y_valid)]
        back_y = back_y[np.logical_and(back_x_valid, back_y_valid)]
        res = mask.shape
        samps_fore = np.random.randint(0, len(fore_x), int(n_samps/2))
        samps_x_fore = fore_x[samps_fore]
        samps_y_fore = fore_y[samps_fore]
        samps_back = np.random.randint(0, len(back_x), n_samps-int(n_samps/2))
        samps_x_back = back_x[samps_back]
        samps_y_back = back_y[samps_back]
        # flip axes to meet properties of 2D keypoints
        pos_fore = np.stack([samps_y_fore/res[1], samps_x_fore/res[0]], axis=1)
        pos_back = np.stack([samps_y_back/res[1], samps_x_back/res[0]], axis=1)

        gt_fore = mask[samps_x_fore, samps_y_fore]
        gt_back = mask[samps_x_back, samps_y_back]

        return np.concatenate([pos_fore, pos_back], axis=0), np.concatenate([gt_fore, gt_back], axis=0)


    def sample_points(self, n_samps, idx, mask, depth, color):
        res = mask.shape
        samps = np.random.randint(0, len(idx[0]), n_samps)
        samps_x = idx[0][samps]
        samps_y = idx[1][samps]
        # flip axes to meet properties of 2D keypoints
        return mask[samps_x, samps_y], \
               depth[samps_x, samps_y], \
               color[samps_x, samps_y, :], \
               np.stack([samps_y/res[1], samps_x/res[0]], axis=1)

    #def sample_points(self, n_samps, idx, mask, std):
    #    res = mask.shape
    #    samps = np.random.randint(0, len(idx[0]), n_samps)
    #    samps_x = idx[0][samps]
    #    samps_y = idx[1][samps]
    #
    #    samps_y01 = samps_y/res[1] + np.random.randn(samps_y.shape[0])*std
    #    samps_x01 = samps_x/res[0] + np.random.randn(samps_x.shape[0])*std
    #    samps_x = np.clip((samps_x01*res[0]).astype(np.int32), 0, res[0]-1)
    #    samps_y = np.clip((samps_y01*res[1]).astype(np.int32), 0, res[1]-1)
    #
    #    return mask[samps_x, samps_y], np.stack([samps_y01, samps_x01], axis=1)

    def __getitem__(self, idx):
        idx = self.steps[idx]

        if self.type == 'pigeons':
            keyPos3d = self.keyPos3d[idx, :, :]
        else:
            keyPos3d = self.keyPos3d[idx-1, :, :]
        onehot = np.eye(self.nkps)
        masks = []
        depth_maps = []
        color_maps = []
        sup_pos = []
        sup_gt = []
        sup_col = []
        sup_dep = []
        sup_pos_foreground = []
        cams = []
        kps_2ds = []

        for j, camera_idx in enumerate(self.cams):
            path = self.path + '{:03d}/'.format(camera_idx)

            intrinsics = self.cameraIntrinsics
            extrinsics = self.cameraExtrinsics[j]
            t_inv = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
            R_inv = extrinsics[:3, :3].T
            ex_inv = np.zeros((3, 4))
            ex_inv[:3, :3] = R_inv
            ex_inv[:3, 3] = t_inv
            trans = self.trans[j]

            # obtain standard sized crop around subject
            radius = self.radius


            camera_params = {'intrinsics': intrinsics, 'extrinsics': extrinsics}




            cams.append({'intrinsics': torch.from_numpy(intrinsics).float(),
                         'extrinsics': torch.from_numpy(extrinsics).float(),
                         'ex_inv': torch.from_numpy(ex_inv).float(),
                         'trans': torch.from_numpy(trans).float()})


        return {'input_pos': torch.from_numpy(keyPos3d).float() / (2*radius),
                'input_feats': torch.from_numpy(onehot).float(),
                'camera_params': cams,
                'frame': idx}


    def get_loader(self, shuffle=True):
        #random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=10, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


#TODO finish up final params!!
def get_synthetic_dataset(data_type, mode, sup_distr, cfg, cams=None):
    return SyntheticDataset(data_type,
                            mode,
                            cfg['training']['npoints_decoder'],
                            sup_distr,
                            cfg['training']['batch_size'],
                            nkps=cfg['data']['nkps'],
                            num_datapoints=cfg['data']['num_datapoints'],
                            radius=cfg['data']['radius'],
                            ncams=cfg['data']['ncams'],
                            cams=cams,
                            normalize_pixel_coords=True,
                            noise_aug=0.003,
                            cam_aug=True
                            )


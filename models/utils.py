import torch
import numpy as np

def proj(camera_params, kps):
    A = torch.bmm(camera_params['intrinsics'], camera_params['extrinsics'])
    tmp = torch.cat([kps, torch.ones([kps.shape[0], kps.shape[1], 1], device=kps.device, dtype=torch.float)], dim=2).permute(0, 2, 1)
    tmp = torch.matmul(A, tmp)
    tmp = tmp.permute(0, 2, 1)
    tmp = tmp / tmp[:, :, 2:]
    return tmp[:, :, :2]

def get2Dkps(camera_params, keyPos3d, mask):
    kps_2d = proj(camera_params, keyPos3d)
    kps_2d[:, :, 0] /= mask.shape[1]
    kps_2d[:, :, 1] /= mask.shape[0]
    return kps_2d


def camera_to_image_plane(pos2d, camera_params):
    intinsics = camera_params['intrinsics'].unsqueeze(1)

    px = pos2d[:, :, 0]
    py = pos2d[:, :, 1]
    w3 = torch.ones_like(px)
    w1 = (px - intinsics[:, :, 0, 2]) / intinsics[:, :, 0, 0]
    w2 = (py - intinsics[:, :, 1, 2]) / intinsics[:, :, 1, 1]

    return torch.stack([w1, w2, w3], dim=-1)


def get_rays_batch(pos2d, camera_params):
    R = camera_params['extrinsics'][:, :3, :3]
    t = camera_params['trans']

    ray_direction_camera = camera_to_image_plane(pos2d, camera_params)

    ray_direction_camera = ray_direction_camera / torch.norm(ray_direction_camera, dim=-1, keepdim=True)

    ray_direction = torch.matmul(torch.inverse(R).unsqueeze(1), ray_direction_camera.unsqueeze(-1))
    ray_origin = t

    return ray_direction, ray_origin


def get_rays(pos2d, camera_params):
    R = camera_params['extrinsics'][:, :3, :3]
    t = camera_params['trans']

    ray_direction_camera = camera_to_image_plane(pos2d, camera_params)

    ray_direction_camera = ray_direction_camera / torch.norm(ray_direction_camera, dim=-1, keepdim=True)

    resOuter = []
    for b in range(R.shape[0]):
        invR = torch.inverse(R[b, ...])
        resInner = []
        for r in range(ray_direction_camera.shape[1]):
            dir = invR @ ray_direction_camera[b, r, :]
            resInner.append(dir)
        resOuter.append(torch.stack(resInner, dim=0))
    ray_direction = torch.stack(resOuter, dim=0)


    #ray_direction = torch.matmul(torch.inverse(R).unsqueeze(1), ray_direction_camera.unsqueeze(-1))
    ray_origin = t

    return ray_direction, ray_origin


def back_proj(camera_params, kps_2d, depth):
    K_inv = np.linalg.inv(camera_params['intrinsics'])
    kps_2d_hom = np.concatenate([kps_2d, np.ones([kps_2d.shape[0], 1])], axis=-1)
    kps_cam = (K_inv @ (kps_2d_hom * depth).T)
    kps = np.linalg.inv(camera_params['extrinsics'][:3, :3]) @ (kps_cam - camera_params['extrinsics'][:, 3:])
    kps = kps.T

    return kps


def back_proj_depth_map(depth, lower_b, camera_params, kps_d, mask):

    threshold = 0.95
    depth[mask > threshold] = depth[mask > threshold]*np.max(kps_d-np.min(kps_d)) + np.min(kps_d)
    forground = np.where(mask > threshold)
    coords_x = forground[1]
    coords_y = forground[0]
    coords = np.stack([coords_x, coords_y], axis=-1)
    depth_values = np.reshape(depth[mask > threshold], (-1, 1))
    p_3d = back_proj(camera_params, coords + lower_b, depth_values)

    return p_3d

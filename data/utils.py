import numpy as np

from scipy.spatial.transform import Rotation as R

def get_intrinsic_params(parser):

    intrinsics = parser['intrinsics']
    pixelsize= np.array([int(intrinsics.get('image_resolution_x_px')), int(intrinsics.get('image_resolution_y_px'))])
    pixelsize = pixelsize / float(intrinsics.get('sensor_size_mm'))

    f = float(intrinsics.get('focal_length_mm')) * np.max(pixelsize)

    c = float(intrinsics.get('sensor_size_mm'))/2. * pixelsize + 0.5

    cameraMatrix = np.array([[f, 0., c[0]],
                             [0., f, c[1]],
                             [0., 0., 1.0]])
    return cameraMatrix


def get_extrinsic_params(parser, cam_id, scale_distance=1, z_offset=0):
    ex_params = parser['extrinsics_Camera.{:03d}'.format(cam_id)]
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        [[1, 0,  0],
         [0, -1, 0],
         [0, 0, -1]])

    sign = np.array([[1., -1., 1.],
                     [-1., 1., -1.],
                     [1., -1, 1.]])


    R_world2bcam = R.from_euler('zyx', (
        ex_params.get('center_cam_rz_rad'), ex_params.get('center_cam_ry_rad'), ex_params.get('center_cam_rx_rad'))).as_matrix()
    R_world2bcam = sign * R_world2bcam


    if scale_distance != 1:
        location = np.array([float(ex_params.get('center_cam_x_m')),
                         float(ex_params.get('center_cam_y_m')),
                         float(ex_params.get('center_cam_z_m'))])
        norm = np.linalg.norm(location)
        #location = (location / norm) * 4.5
        location = location * scale_distance
        location[-1] += z_offset
        print('Att: scaling cameras to sphere')
    else:
        location = np.array([float(ex_params.get('center_cam_x_m'))*scale_distance,
                             float(ex_params.get('center_cam_y_m'))*scale_distance,
                             float(ex_params.get('center_cam_z_m'))*scale_distance+z_offset])


    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1.0 * R_world2bcam @ location

    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    RT = np.concatenate([R_world2cv, np.expand_dims(T_world2cv, 1)], axis=1)
    ###RT = np.concatenate([rotm, np.expand_dims(location, 1)], axis=1)
    return RT, R_world2cv, location

def proj(camera_params, kps):
    A = camera_params['intrinsics'] @ camera_params['extrinsics']

    tmp = A @ np.concatenate([kps, np.ones([kps.shape[0], 1])], axis=1).T
    tmp = tmp.T

    tmp = tmp / tmp[:, 2:]
    return tmp[:, :2]

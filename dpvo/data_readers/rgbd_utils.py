import numpy as np
import os.path as osp

import torch
from ..lietorch import SE3

from scipy.spatial.transform import Rotation

def parse_list(filepath, skiprows=0):
    """ read list data """
    data = np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)
    return data

def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=1.0):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j = np.argmin(np.abs(tstamp_depth - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))
        
            if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                    (np.abs(tstamp_pose[k] - t) < max_dt):
                associations.append((i, j, k))
            
    return associations

def loadtum(datapath, frame_rate=-1):
    """ read video data in tum-rgbd format """
    if osp.isfile(osp.join(datapath, 'groundtruth.txt')):
        pose_list = osp.join(datapath, 'groundtruth.txt')
    
    elif osp.isfile(osp.join(datapath, 'pose.txt')):
        pose_list = osp.join(datapath, 'pose.txt')

    else:
        return None, None, None, None

    image_list = osp.join(datapath, 'rgb.txt')
    depth_list = osp.join(datapath, 'depth.txt')

    calib_path = osp.join(datapath, 'calibration.txt')
    intrinsic = None
    if osp.isfile(calib_path):
        intrinsic = np.loadtxt(calib_path, delimiter=' ')
        intrinsic = intrinsic.astype(np.float64)

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:,1:].astype(np.float64)

    tstamp_image = image_data[:,0].astype(np.float64)
    tstamp_depth = depth_data[:,0].astype(np.float64)
    tstamp_pose = pose_data[:,0].astype(np.float64)
    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

    # print(len(tstamp_image))
    # print(len(associations))

    indicies = range(len(associations))[::5]

    # indicies = [ 0 ]
    # for i in range(1, len(associations)):
    #     t0 = tstamp_image[associations[indicies[-1]][0]]
    #     t1 = tstamp_image[associations[i][0]]
    #     if t1 - t0 > 1.0 / frame_rate:
    #         indicies += [ i ]

    images, poses, depths, intrinsics, tstamps = [], [], [], [], []
    for ix in indicies:
        (i, j, k) = associations[ix]
        images += [ osp.join(datapath, image_data[i,1]) ]
        depths += [ osp.join(datapath, depth_data[j,1]) ]
        poses += [ pose_vecs[k] ]
        tstamps += [ tstamp_image[i] ]
        
        if intrinsic is not None:
            intrinsics += [ intrinsic ]

    return images, depths, poses, intrinsics, tstamps


def all_pairs_distance_matrix(poses, beta=2.5):
    """ compute distance matrix between all pairs of poses """
    poses = np.array(poses, dtype=np.float32)
    poses[:,:3] *= beta # scale to balence rot + trans
    poses = SE3(torch.from_numpy(poses))

    r = (poses[:,None].inv() * poses[None,:]).log()
    return r.norm(dim=-1).cpu().numpy()

def pose_matrix_to_quaternion(pose):
    """ convert 4x4 pose matrix to (t, q) """
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate([pose[:3, 3], q], axis=0)


def compute_distance_matrix_flow(poses, disps, intrinsics):
    """ compute flow magnitude between all pairs of frames """
    if not isinstance(poses, SE3):
        poses = torch.from_numpy(poses).float().cuda()[None]
        poses = SE3(poses).inv()

        disps = torch.from_numpy(disps).float().cuda()[None]
        intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1).cuda()
    jj = jj.reshape(-1).cuda()

    MAX_FLOW = 100.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        # flow1, val1 = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        # flow2, val2 = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s])
        flow1, val1 = induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2, val2 = induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s])


        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.7] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix


def compute_distance_matrix_flow2(poses, disps, intrinsics, beta=0.4):
    """ compute flow magnitude between all pairs of frames """
    # if not isinstance(poses, SE3):
    #     poses = torch.from_numpy(poses).float().cuda()[None]
    #     poses = SE3(poses).inv()

    #     disps = torch.from_numpy(disps).float().cuda()[None]
    #     intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    MAX_FLOW = 128.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1a, val1a = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s], tonly=True)
        flow1b, val1b = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2a, val2a = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s], tonly=True)
        flow2b, val2b = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])

        flow1 = flow1a + beta * flow1b
        val1 = val1a * val2b

        flow2 = flow2a + beta * flow2b
        val2 = val2a * val2b
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.8] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix










MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)

def iproj(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """ map points from ii->jj """

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    
    # transform
    Gij = poses[:,jj] * poses[:,ii].inv()

    Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian)
    
    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def induced_flow(poses, disps, intrinsics, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[...,:2] - coords0, valid

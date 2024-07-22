import re
import cv2
import glob
import os
import datetime
import numpy as np
import os.path as osp
import pandas as pd
import math
import os.path

from pathlib import Path
from scipy.spatial.transform import Rotation as R

import torch
from dpvo.dpvo import DPVO
from dpvo.utils import *
from dpvo.config import cfg

from dpvo.data_readers.tartan import test_split as val_split
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

test_split = [
    'Bosch06_EP01', 'Bosch06_EP02', 'Bosch06_EP04', 'Bosch06_EP05',
    'Bosch06_EP06', 'Bosch06_EP07', 'Bosch06_EP08', 'Bosch07_EP01',
    'Bosch07_EP02', 'Bosch07_EP03', 'Bosch07_EP04', 'Bosch07_EP05',
    'Bosch07_EP06', 'Bosch07_EP07', 'Bosch07_EP08'
]

# test_split = ['Bosch06_EP01']

STRIDE = 1
fx, fy, cx, cy = [320, 320, 320, 240]


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


def quaternion_to_rotation_vector(q):
    """
    Convert a quaternion to a rotation vector (axis-angle representation).
    
    Parameters:
    q (np.ndarray): Quaternion as a 4-element array [w, x, y, z]
    
    Returns:
    np.ndarray: Rotation vector as a 3-element array [rx, ry, rz]
    """
    w, x, y, z = q
    # Compute the angle
    angle = 2 * np.arccos(w)

    # Compute the axis
    s = np.sqrt(1 - w * w)  # Sine of half the angle

    # To avoid division by zero
    if s < 1e-8:
        axis = np.array([x, y, z])
    else:
        axis = np.array([x, y, z]) / s

    # Rotation vector
    rotation_vector = angle * axis

    return rotation_vector


def rotation_matrix_to_quaternion(R):
    q = np.empty((4, ))
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    return q


# Function to convert a quaternion back to a rotation matrix
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.array([[
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w
    ], [
        2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w
    ], [
        2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y
    ]])
    return R


def is_orthogonal(matrix, tol=1e-6):
    I = np.eye(3)
    should_be_I = np.dot(matrix.T, matrix)
    return np.allclose(should_be_I, I, atol=tol)


def get_ref_traj(shot_dir):
    camerasdir = os.path.join(shot_dir, "camera.pkl")
    cameras_dict = pd.read_pickle(camerasdir)
    sorted_cameras_dict = dict(sorted(cameras_dict.items()))

    imagedir = os.path.join(shot_dir, "frames")
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(".jpg")))
    imids = [im_id.split('.jpg')[0].split('_')[-1] for im_id in imfiles]

    # Adjust initialization to account for stride
    traj_ref_np_len = math.ceil(len(imids) / STRIDE)
    traj_ref_np = np.zeros((traj_ref_np_len, 7))
    # for timestamp, (key, value) in enumerate(sorted_cameras_dict.items())[::STRIDE]:
    timestamp = 0
    for _, (key,
            value) in enumerate(sorted(sorted_cameras_dict.items())[::STRIDE]):
        if key in imids:
            v2c = value['v2c']
            rotation_matrix = R.from_matrix(v2c[:3, :3])
            translation = v2c[:3, 3]
            quaternion = rotation_matrix.as_quat()
            value['tum'] = {
                'quaternion': quaternion,
                'translation': translation
            }
            traj_ref_np[timestamp][0:3] = translation
            traj_ref_np[timestamp][3:] = quaternion
            timestamp = timestamp + 1

    return traj_ref_np


def video_iterator(shot_dir, ext=".jpg", preload=True):

    imagedir = os.path.join(shot_dir, "frames")
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(ext)))

    camerasdir = os.path.join(shot_dir, "camera.pkl")
    cameras_dict = pd.read_pickle(camerasdir)

    data_list = []
    for imfile in sorted(imfiles)[::STRIDE]:
        image_numpy = cv2.imread(imfile)

        image = torch.from_numpy(image_numpy).permute(2, 0, 1)
        image_id = imfile.split('.jpg')[0].split('_')[-1]
        camera_intrins = cameras_dict[image_id]['intr_mtx']
        fx = camera_intrins[0, 0]
        fy = camera_intrins[1, 1]
        cx = camera_intrins[0, 2]
        cy = camera_intrins[1, 2]
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image, intrinsics, image_numpy))

    if len(data_list) == 0: raise Exception("Failed to read input.")

    for (image, intrinsics, image_numpy) in data_list:
        yield image.cuda(), intrinsics.cuda(), image_numpy


@torch.no_grad()
def run(imagedir, cfg, network, viz=False):
    slam = DPVO(cfg, network, ht=720, wd=1280, viz=viz)

    for t, (image, intrinsics,
            image_numpy) in enumerate(video_iterator(imagedir)):
        if viz:
            show_image(image, 1)

        with Timer("SLAM", enabled=viz):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    return slam.terminate()


def ate(traj_ref, traj_est, timestamps):
    import evo
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation

    traj_est = PoseTrajectory3D(positions_xyz=traj_est[:, :3],
                                orientations_quat_wxyz=traj_est[:, 3:],
                                timestamps=timestamps)

    traj_ref = PoseTrajectory3D(positions_xyz=traj_ref[:, :3],
                                orientations_quat_wxyz=traj_ref[:, 3:],
                                timestamps=timestamps)

    result = main_ape.ape(traj_ref,
                          traj_est,
                          est_name='traj',
                          pose_relation=PoseRelation.translation_part,
                          align=True,
                          correct_scale=True)

    return result.stats["rmse"]


@torch.no_grad()
def evaluate(config,
             net,
             split="validation",
             trials=1,
             plot=False,
             save=False):

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    if not os.path.isdir("AmazonResults"):
        os.mkdir("AmazonResults")

    scenes = test_split
    print("scenes ", scenes)

    results = {}
    # shot_path = '/home/oem/Rembrand/moving_camera/datasets/Amazon/small_baseline_dataset/Bosch07_EP01/Bosch07_01_SHOT_018'
    # traj_ref = get_ref_traj(shot_path)
    # print("traj_ref ", traj_ref.shape)

    # torch.cuda.empty_cache()
    # traj_est, tstamps = run(shot_path, config, net)
    # torch.cuda.empty_cache()
    # print("traj_est ", traj_est.shape)
    # ate_score = ate(traj_ref, traj_est, tstamps)
    # print("ate_score ", ate_score)
    for i, scene in enumerate(scenes):
        scene_path = os.path.join(
            "/home/oem/Rembrand/moving_camera/datasets/small_baseline_dataset",
            scene)
        shots = os.listdir(scene_path)
        for s, shot in enumerate(shots):

            shot_results = []
            for j in range(trials):
                shot_path = os.path.join(scene_path, shot)

                file_path= f"AMAZON_saved_trajectories/AMAZON_{shot}_Trial{j+1:02d}.txt"
                if os.path.exists(file_path):
                    continue

                print("---->> shot : ", shot)

                traj_ref = get_ref_traj(shot_path)
                print("traj_ref ", traj_ref.shape)

                torch.cuda.empty_cache()
                traj_est, tstamps = run(shot_path, config, net)
                print("traj_est ", traj_est.shape)

                torch.cuda.empty_cache()
                # ate_score = ate(traj_ref, traj_est, tstamps)
                # print("ate_score  ",ate_score)

                print("------------------------------ ")

                # shot_results.append(ate_score)
                if plot:

                    Path("AMAZON_trajectory_plots").mkdir(exist_ok=True)
                    plot_trajectory(
                        (traj_est, tstamps), (traj_ref, tstamps),
                        f"AMAZON {shot.replace('_', ' ')} Trial #{j+1} (ATE: {ate_score:.03f})",
                        f"AMAZON_trajectory_plots/AMAZON_{scene}_{shot}_Trial{j+1:02d}.pdf",
                        align=True,
                        correct_scale=True)
                if save:
                    Path("AMAZON_saved_trajectories").mkdir(exist_ok=True)
                    save_trajectory_tum_format((
                        traj_est, tstamps
                    ), f"AMAZON_saved_trajectories/AMAZON_{shot}_Trial{j+1:02d}.txt"
                                               )
                    save_trajectory_tum_format((
                        traj_ref, np.arange(0, traj_ref.shape[0] + 1)
                    ), f"AMAZON_saved_trajectories/AMAZON_ref_{shot}_Trial{j+1:02d}.txt"
                                               )

            results[shot] = np.median(shot_results)

    return results


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--id', type=int, default=-1)
    parser.add_argument('--weights', default="dpvo.pth")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(1234)

    evaluate(cfg,
             args.weights,
             split=args.split,
             trials=args.trials,
             plot=args.plot,
             save=args.save_trajectory)

    # results = evaluate(cfg,
    #                    args.weights,
    #                    split=args.split,
    #                    trials=args.trials,
    #                    plot=args.plot,
    #                    save=args.save_trajectory)
    # for k in results:
    #     print(k, results[k])

import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from ..lietorch import SE3
from .base import RGBDDataset

# cur_path = osp.dirname(osp.abspath(__file__))
# test_split = osp.join(cur_path, 'tartan_test.txt')
# test_split = open(test_split).read().split()

test_split = [
    "abandonedfactory/abandonedfactory/Easy/P011",
    "abandonedfactory/abandonedfactory/Hard/P011",
    "abandonedfactory_night/abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/abandonedfactory_night/Hard/P014",
    "amusement/amusement/Easy/P008",
    "amusement/amusement/Hard/P007",
    "carwelding/carwelding/Easy/P007",
    "endofworld/endofworld/Easy/P009",
    "gascola/gascola/Easy/P008",
    "gascola/gascola/Hard/P009",
    "hospital/hospital/Easy/P036",
    "hospital/hospital/Hard/P049",
    "japanesealley/japanesealley/Easy/P007",
    "japanesealley/japanesealley/Hard/P005",
    "neighborhood/neighborhood/Easy/P021",
    "neighborhood/neighborhood/Hard/P017",
    "ocean/ocean/Easy/P013",
    "ocean/ocean/Hard/P009",
    "office2/office2/Easy/P011",
    "office2/office2/Hard/P010",
    "office/office/Hard/P007",
    "oldtown/oldtown/Easy/P007",
    "oldtown/oldtown/Hard/P008",
    "seasidetown/seasidetown/Easy/P009",
    "seasonsforest/seasonsforest/Easy/P011",
    "seasonsforest/seasonsforest/Hard/P006",
    "seasonsforest_winter/seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/seasonsforest_winter/Hard/P018",
    "soulcity/soulcity/Easy/P012",
    "soulcity/soulcity/Hard/P009",
    "westerndesert/westerndesert/Easy/P013",
    "westerndesert/westerndesert/Hard/P007",
]

# test_split = [
#     "abandonedfactory_sample_P001/P001",
#     "abandonedfactory_night_sample_P002/P002",
#     "amusement_sample_P008/P008",
# ]

exist_scenes = [
    'abandonedfactory/abandonedfactory/Easy/P001',
    'abandonedfactory_night/abandonedfactory_night/Easy/P002',
    'amusement/amusement/Easy/P008', 'carwelding/carwelding/Easy/P007',
    'endofworld/endofworld/Easy/P001', 'gascola/gascola/Easy/P001',
    'hospital/hospital/Easy/P000', 'japanesealley/japanesealley/Easy/P007',
    'neighborhood/neighborhood/Easy/P002', 'ocean/ocean/Easy/P006',
    'office2/office2/Easy/P003', 'seasidetown/seasidetown/Easy/P003',
    'seasonsforest/seasonsforest/Easy/P002',
    'seasonsforest_winter/seasonsforest_winter/Easy/P006',
    'soulcity/soulcity/Easy/P003', 'westerndesert/westerndesert/Easy/P002'
]


class TartanAir(RGBDDataset):
    ''' inherits from RGBDDataset '''

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAir, self).__init__(name='TartanAir', **kwargs)

    @staticmethod
    def is_test_scene(scene):
        def modify_path(path):
            parts = path.split('/')
            return '/'.join(parts[1:])

        return any(modify_path(x) in scene for x in test_split)

    @staticmethod
    def is_scene_found(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in exist_scenes)

    def _build_dataset(self):
        ''' Construct the dataset by loading images, depths, poses, and intrinsics, and creating a frame co-visible graph'''
        
        from tqdm import tqdm
        print("Building TartanAir dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/*/'))
        print("scenes ",scenes)
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image_left/*.png'))) # all image files for each scene
            depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy'))) # all depth files for each scene

            # Number of images must equal Number of depth maps in order to add the scene
            if len(images) != len(depths):
                continue
            
            # Load poses of the scene 
            poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            # Reorder the columns of the poses array -> (x,y,z,q_x, q_y, q_z, q_w)
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            # Scale the first three columns of the poses array by dividing them by TartanAir.DEPTH_SCALE.
            poses[:, :3] /= TartanAir.DEPTH_SCALE

            # list of intrinsic camera parameters for each image in the dataset
            intrinsics = [TartanAir.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {
                'images': images,
                'depths': depths,
                'poses': poses,
                'intrinsics': intrinsics,
                'graph': graph
            }

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth == np.nan] = 1.0
        depth[depth == np.inf] = 1.0
        return depth

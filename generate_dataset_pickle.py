import pickle
import os
import os.path as osp

# RGBD-Dataset
from dpvo.data_readers.tartan import TartanAir

dataset = TartanAir(
    datapath="/mnt/data/visual_slam/tartanair/")
scene_info = dataset._build_dataset()

with open('tartanair.pickle', 'wb') as handle:
    pickle.dump(scene_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

scene_info__ = pickle.load(open('tartanair.pickle', 'rb'))

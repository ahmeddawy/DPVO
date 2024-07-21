import pickle
import os
import os.path as osp

# RGBD-Dataset
from dpvo.data_readers.tartan import TartanAir

dataset = TartanAir(
    datapath="/home/oem/Rembrand/moving_camera/datasets/TartanAir")
scene_info = dataset._build_dataset()
print("scene_info ",type(scene_info))

with open('my.pickle', 'wb') as handle:
    pickle.dump(scene_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

scene_info__ = pickle.load(open('my.pickle', 'rb'))

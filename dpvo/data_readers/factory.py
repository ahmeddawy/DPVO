import pickle
import os
import os.path as osp

# RGBD-Dataset
from .tartan import TartanAir


def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    #  Dict to map dataset names ('tartan') to their corresponding dataset classes (TartanAir).
    dataset_map = {
        'tartan': (TartanAir, ),
    }
    ''' Iterate over the dataset_list, and for each dataset name,
      Create an instance of the corresponding dataset class, passing any additional keyword arguments'''
    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)
        print("Dataset {} has {} images".format(key, len(db)))

        # Collect all the created datasets in db_list and then concatenates them into a single dataset using ConcatDataset.
        db_list.append(db)

    return ConcatDataset(db_list)

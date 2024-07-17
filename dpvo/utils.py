import torch
import torch.nn.functional as F
from ultralytics import YOLO
import cv2
import numpy as np

import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

all_times = []


class Timer:

    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()

    def __exit__(self, type, value, traceback):
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end)
            all_times.append(elapsed)
            print(self.name, elapsed)


def coords_grid(b, n, h, w, **kwargs):
    """ coordinate grid """
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)
    coords = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    return coords[[1, 0]].view(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)


def coords_grid_with_index(d, **kwargs):
    """ coordinate grid with frame index"""
    b, n, h, w = d.shape
    i = torch.ones_like(d)
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords = torch.stack([x, y, d], dim=2)
    index = torch.arange(0, n, dtype=torch.float, **kwargs)
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index


def patchify(x, patch_size=3):
    """ extract patches from video """
    b, n, c, h, w = x.shape
    x = x.view(b * n, c, h, w)
    y = F.unfold(x, patch_size)
    y = y.transpose(1, 2)
    return y.reshape(b, -1, c, patch_size, patch_size)


def pyramidify(fmap, lvls=[1]):
    """ turn fmap into a pyramid """
    b, n, c, h, w = fmap.shape

    pyramid = []
    for lvl in lvls:
        gmap = F.avg_pool2d(fmap.view(b * n, c, h, w), lvl, stride=lvl)
        pyramid += [gmap.view(b, n, c, h // lvl, w // lvl)]

    return pyramid


def all_pairs_exclusive(n, **kwargs):
    ii, jj = torch.meshgrid(torch.arange(n, **kwargs),
                            torch.arange(n, **kwargs))
    k = ii != jj
    return ii[k].reshape(-1), jj[k].reshape(-1)


def set_depth(patches, depth):
    patches[..., 2, :, :] = depth[..., None, None]
    return patches


def flatmeshgrid(*args, **kwargs):
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid)


def get_human_masks(image, human_segmentor):
    kernel = np.ones((21, 21), np.uint8)
    all_human_mask_pixels = None
    if image is not None:
        human_mask_results = human_segmentor(image, conf=0.5, verbose=False , device='cpu')
        human_mask_pixels = []

        # Process each result
        for data in human_mask_results:
            classes = data.boxes.cls.tolist()
            names = data.names
            if data.masks is not None:
                masks = data.masks.xy
                for i, mask in enumerate(masks):
                    class_name = names[int(classes[i])]
                    if class_name == "person":
                        # Create a mask image
                        mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask_img,
                                     [np.array(mask, dtype=np.int32)], 255)

                        dilated_mask_img = cv2.dilate(mask_img,
                                                      kernel,
                                                      iterations=1)

                        # Store the pixels of the mask
                        mask_pixels = np.column_stack(
                            np.where(dilated_mask_img == 255))
                        # mask_pixels = mask_pixels // 4
                        human_mask_pixels.append(mask_pixels)
        all_human_mask_pixels = np.vstack(
            human_mask_pixels) if human_mask_pixels else np.array([])
    return all_human_mask_pixels


def get_disparity(image, depth_anything):
    # input_size = 518
    # depth = depth_anything.infer_image(image, input_size)
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # min_depth_threshold = 1e-6
    # depth[depth < min_depth_threshold] = min_depth_threshold
    # disparity_map = (1 * 1) / depth
    # depth = depth.astype(np.uint8)
    # depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    # disparity_map = disparity_map.astype(np.float32)

    depth = depth_anything.infer_image(
        image)  # HxW depth map in meters in numpy

    depth[depth < 0.01] = np.mean(depth)

    disparity_map = (1 * 1) / depth

    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

    return depth, disparity_map

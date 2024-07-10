import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from multiprocessing import Process, Queue

from dpvo.utils import *
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream

SKIP = 0


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


@torch.no_grad()
def run(cfg,
        network,
        imagedir,
        calib,
        stride=1,
        skip=0,
        viz=False,
        timeit=False):
    torch.cuda.empty_cache()
    slam = None
    queue = Queue(maxsize=8)

    human_segmentor = YOLO('yolov8n-seg.pt')
    human_segmentor.to(torch.device("cpu"))
    kernel = np.ones((41, 41), np.uint8)

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream,
                         args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream,
                         args=(queue, imagedir, calib, stride, skip))

    reader.start()

    while 1:
        torch.cuda.empty_cache()
        (t, image, intrinsics) = queue.get()

        all_human_mask_pixels = get_human_masks(image)

        if t < 0: break
        # the raw input image is resized to its 0.5 scale in video_stream
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()

        # the intrinsics shape in 4, intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg,
                        network,
                        ht=image.shape[1],
                        wd=image.shape[2],
                        viz=viz)

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics,
                 human_masks=all_human_mask_pixels)  # calling  __call__

    for _ in range(12):
        slam.update()

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip,
        args.viz, args.timeit)

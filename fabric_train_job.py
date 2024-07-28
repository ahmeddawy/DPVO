import cv2
import os
import argparse
import numpy as np
import neptune

from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dpvo.data_readers.factory import dataset_factory

from dpvo.lietorch import SE3
from dpvo.logger import Logger
import torch.nn.functional as F

from dpvo.net import VONet
from evaluate_tartan import evaluate as validate
from lightning.fabric import Fabric

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)
    torch.cuda.empty_cache()
    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)
    torch.cuda.empty_cache()
    c = VarA / torch.trace(torch.diag(D))
    return c


def train_step(fabric, net, optimizer, scheduler, data_blob, so):
    
    images, poses, disps, intrinsics = [
        x.float() for x in data_blob
    ]
    optimizer.zero_grad()

    poses = SE3(poses).inv()
    traj = net(images,
                poses,
                disps,
                intrinsics,
                M=1024,
                STEPS=8,
                structure_only=so)

    loss = 0.0
    flow_loss = 0.0
    pose_loss = 0.0
    for i, (v, x, y, P1, P2, kl) in enumerate(traj):
        e = (x - y).norm(dim=-1)
        e = e.reshape(-1, 9)[(v > 0.5).reshape(-1)].min(dim=-1).values

        N = P1.shape[1]
        ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        ii = ii.reshape(-1).cuda()
        jj = jj.reshape(-1).cuda()

        k = ii != jj
        ii = ii[k]
        jj = jj[k]

        P1 = P1.inv()
        P2 = P2.inv()

        t1 = P1.matrix()[..., :3, 3]
        t2 = P2.matrix()[..., :3, 3]

        s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
        P1 = P1.scale(s.view(1, 1))

        dP = P1[:, ii].inv() * P1[:, jj]
        dG = P2[:, ii].inv() * P2[:, jj]

        e1 = (dP * dG.inv()).log()
        tr = e1[..., 0:3].norm(dim=-1)
        ro = e1[..., 3:6].norm(dim=-1)

        loss += args.flow_weight * e.mean()
        flow_loss += e.mean()

        if not so and i >= 2:
            loss += args.pose_weight * (tr.mean() + ro.mean())
            pose_loss +=  (tr.mean() + ro.mean())

    # kl is 0 (not longer used)
    loss += kl

    fabric.backward(loss)
    fabric.clip_gradients(net, optimizer, max_norm=0.5)

    optimizer.step()
    scheduler.step()

    metrics = {
        "loss": loss.item(),
        "kl": kl.item(),
        "px1": (e < .25).float().mean().item(),
        "ro": ro.float().mean().item(),
        "tr": tr.float().mean().item(),
        "r1": (ro < .001).float().mean().item(),
        "r2": (ro < .01).float().mean().item(),
        "t1": (tr < .001).float().mean().item(),
        "t2": (tr < .01).float().mean().item(),
    }
    return loss, flow_loss, pose_loss, metrics

def val_step(fabric, net, optimizer,):
    pass

def train(args):
    """ main training loop """
    fabric = Fabric(strategy="ddp", accelerator="cuda")
    fabric.launch()

    # run = neptune.init_run(
    #     project="Rembrand/mc-dpvo",
    #     api_token=
    #     "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNzA0MGVmZC0zM2VjLTQwYjAtYWNhNC0wYTg1OTJmMGRkZjcifQ==",
    # )

    # params = {
    #     "experiment_name": args.name,
    #     "learning_rate": args.lr,
    #     "optimizer": "AdamW",
    #     "scheduler": "OneCycleLR",
    #     "training steps": args.steps,
    # }

    # run["parameters"] = params

    db = dataset_factory(['tartan'],
                         datapath="/mnt/data/visual_slam/tartanair",
                         n_frames=args.n_frames)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=8)

    net = VONet()

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict, strict=False)

    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                args.lr,
                                                args.steps*len(train_loader),
                                                pct_start=0.01,
                                                cycle_momentum=False,
                                                anneal_strategy='linear')
    
    net, optimizer = fabric.setup(net, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)
    if fabric.global_rank == 0:
        logger = Logger(args.name, scheduler)

    total_steps = 0

    net.train()

    while 1:
        for data_blob in train_loader:
            # fix poses to gt for first 1k steps
            so = total_steps < 1000 and args.ckpt is None

            loss, flow_loss, pose_loss, metrics = \
                train_step(fabric, net, optimizer, scheduler, data_blob, so)
            
            # neptune logging
            # run["train/loss"].append(loss)
            # run["train/flow_loss"].append(flow_loss)
            # run["train/pose_loss"].append(pose_loss)
            # run["metrics"].append(metrics)

            # stdout logging
            if fabric.global_rank == 0:
                logger.push(metrics)
            total_steps += 1


            if total_steps % 10000 == 0:
                # torch.cuda.empty_cache()

                net.eval()

                if fabric.global_rank == 0:
                    PATH = '/DPVO/training_checkpoints/%s_%06d.pth' % (
                        args.name, total_steps)
                    torch.save(net.state_dict(), PATH)
                fabric.barrier()
                
                # validation_results = validate(None, net)
                # run["validation_results"].append(validation_results)

                # if fabric.global_rank == 0:
                #     logger.write_dict(validation_results)

                # torch.cuda.empty_cache()
                net.train()

            if  total_steps > args.steps :
                break

        break
    # run.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    args = parser.parse_args()

    train(args)

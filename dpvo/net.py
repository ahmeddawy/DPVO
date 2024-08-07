import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

DIM = 384


def get_random_pixels_not_in_mask(w, h, mask_pixels, patches_per_image,
                                  device):
    x_coords = []
    y_coords = []

    while len(x_coords) < patches_per_image:
        x = torch.randint(1,
                          w - 1,
                          size=[patches_per_image - len(x_coords)],
                          device=device)
        y = torch.randint(1,
                          h - 1,
                          size=[patches_per_image - len(y_coords)],
                          device=device)

        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()

        valid_indices = [(i, j) for i, j in zip(x_cpu, y_cpu)
                         if not ((j, i) in mask_pixels)]

        if valid_indices:
            x_coords.extend([i for i, j in valid_indices])
            y_coords.extend([j for i, j in valid_indices])

    return torch.tensor(
        x_coords[:patches_per_image],
        device=device).unsqueeze(0), torch.tensor(y_coords[:patches_per_image],
                                                  device=device).unsqueeze(0)


class Update(nn.Module):

    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True),
                                nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True),
                                nn.Linear(DIM, DIM))

        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2 * 49 * p * p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2),
                               GradientClip())

        self.w = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2),
                               GradientClip(), nn.Sigmoid())

    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """
        # Inputs shapes
        # net -> Size[1,len(ii),384]
        # inp are the patches context maps -> Size[1,Number of Patches,384]
        # corr are the cooleration between patches[kk] feature maps and frame_jj feature map -> Size[1,Number of Patches,882]
        # ii are index of the frame that patches[kk] belong to -> Size[1,Number of Patches]
        # jj are index of frame to which patches[kk] are projected to -> Size[1,Number of Patches]
        # kk are indices of the patches -> Size[1,Number of Patches]

        # Step-1
        # a- transfom coorelation from 882 to 384 using self.corr -> size[batch_size, Number of Patches,384 ]
        # b- net[1,Number of Patches,384] = net[1,Number of Patches,384] + inp(patches context maps)[1,Number of Patches,384]+ trasformed coor[]
        # c- normalize net
        net = net + inp + self.corr(
            corr)  # --> correlation calc from the paper
        net = self.norm(net)
        #-------------------------------------------------------------------------------------------------------#

        # Step-2
        # a- Find the neighboring indices for patches_kk and frames_jj
        # b- Create a mask for valid indices ix , jx
        # c- The network state is updated by aggregating information from neighboring patches and frames using the learned transformations c1 and c2
        ''' - If SLAM is not init, we sends only the patches and the frames to which it is projects not the complete graph.
            -- Thus, there will be no neighbors.
            -- mask_ix * net[:, ix]-> and  mask_jx * net[:, jx] are all zeros 
        '''

        ix, jx = fastba.neighbors(
            kk, jj)  # --> This contains the 1D convolution  from the paper

        mask_ix = (ix >= 0).float().reshape(
            1, -1, 1
        )  #reshapes the masks to have the shape [1, length_of_valid_indices, 1].
        mask_jx = (jx >= 0).float().reshape(
            1, -1, 1
        )  #reshapes the masks to have the shape [1, length_of_valid_indices, 1].
        '''
           (mask_ix * net[:, ix]) -> From the net state, select the states of the valid previous neighbors. For the not valid previous neighbors,
           their contribution will be suppressed to zero by the mask 
        
        '''
        net = net + self.c1(mask_ix * net[:, ix])
        net = net + self.c2(mask_jx * net[:, jx])
        #-------------------------------------------------------------------------------------------------------#

        net = net + self.agg_kk(net, kk)  # --> SoftAgg from the paper
        net = net + self.agg_ij(net, ii * 12345 + jj)

        net = self.gru(net)  # --> Transition block from the paper

        # --> self.d and self.w are the factor head the paper
        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):

    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]
        dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self,
                images,
                patches_per_image=80,
                disps=None,
                gradient_bias=False,
                return_color=False,
                human_masks=None):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0  # size(1, 1, 128, 132, 240)
        imap = self.inet(images) / 4.0  # size(1, 1, 128, 132, 240)

        b, n, c, h, w = fmap.shape
        P = self.patch_size
        # bias patch selection towards regions with high gradient
        if gradient_bias:
            g = self.__image_gradient(images)
            x = torch.randint(1,
                              w - 1,
                              size=[n, 3 * patches_per_image],
                              device="cuda")
            y = torch.randint(1,
                              h - 1,
                              size=[n, 3 * patches_per_image],
                              device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0, :, None], coords,
                                 0).view(n, 3 * patches_per_image)

            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        else:
            if human_masks is None:
                x = torch.randint(1,
                                  w - 1,
                                  size=[n, patches_per_image],
                                  device="cuda")
                y = torch.randint(1,
                                  h - 1,
                                  size=[n, patches_per_image],
                                  device="cuda")
            else:
                # x = torch.randint(1,
                #                   w - 1,
                #                   size=[n, patches_per_image],
                #                   device="cuda")
                # y = torch.randint(1,
                #                   h - 1,
                #                   size=[n, patches_per_image],
                #                   device="cuda")
                human_masks = human_masks // 4
                mask_pixels_set = set(map(tuple, human_masks))

                x, y = get_random_pixels_not_in_mask(w,
                                                     h,
                                                     mask_pixels_set,
                                                     patches_per_image,
                                                     device="cuda")

        # we have the center coordinates (x,y) of the patches
        coords = torch.stack([x, y], dim=-1).float()  # size (1,96,2)

        # Extract the corresponding patches from the context map and the feature map

        # Extract the corresponding patches from the context map centered around each coordinate in coords with radious 0,
        # The extracted patches are then reshaped to have the shape [b, -1, 384, 1, 1],
        imap = altcorr.patchify(imap[0], coords,
                                0).view(b, -1, DIM, 1,
                                        1)  # Size([1, 96, 384, 1, 1])

        # Extract the corresponding patches from the features map centered around each coordinate in coords with radious 1,
        # The extracted patches are then reshaped to have the shape [b, -1, 128, 3, 3],
        gmap = altcorr.patchify(fmap[0], coords,
                                P // 2).view(b, -1, 128, P,
                                             P)  # Size([1, 96, 128, 3, 3])

        if return_color:
            clr = altcorr.patchify(images[0], 4 * (coords + 0.5),
                                   0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(
            disps, device=fmap.device
        )  # Size([1, 1, 3, 132, 240]) this grid is (x,y,d) for each pixel of the disparity

        # Extracting patches of size P x P centered around each coordinate in coords from the grid tensor,
        #  The extracted patches are then reshaped to have the shape [b, -1, 3, P, P],
        #  where b is the batch size,
        #  -1 represents the number of patches,
        #  3 is the number of channels (x, y, disparity),
        #  and P x P is the patch size.
        patches = altcorr.patchify(grid[0], coords,
                                   P // 2).view(b, -1, 3, P,
                                                P)  # Size([1, 96, 3, 3, 3]) ,

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        # return
        # 1- feature map Size(1, 1, 128, 132, 240)
        # 2- corresponding patches of the feature map Size([1, 96, 128, 3, 3])
        # 3- corresponding patches of the context map Size([1, 96, 384, 1, 1])
        # 4- patches of the image centered around coords with added disparity Size([1, 96, 3, 3, 3])

        return fmap, gmap, imap, patches, index


class CorrBlock:

    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1, 4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [
                altcorr.corr(self.gmap, self.pyramid[i],
                             coords / self.levels[i], ii, jj, self.radius,
                             self.dropout)
            ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):

    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4

    @autocast(enabled=False)
    def forward(self,
                images,
                poses,
                disps,
                intrinsics,
                M=1024,
                STEPS=12,
                P=1,
                structure_only=False,
                rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float() # Downsample

        # 1- feature map of the frame Size(1, 1, 128, 132, 240)
        # 2- corresponding patches of the feature map Size([1, 96, 128, 3, 3])
        # 3- corresponding patches of the context map Size([1, 96, 384, 1, 1])
        # 4- patches of the image centered around coords with added disparity Size([1, 96, 3, 3, 3])
        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap) 

        b, N, c, h, w = fmap.shape # Size(1, 1, 128, 132, 240)
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p // 2, p // 2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(
            torch.where(ix < 8)[0], torch.arange(0, 8, device="cuda"))
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)

        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]

        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:, n] = Gs.data[:, n - 1]
                kk1, jj1 = flatmeshgrid(
                    torch.where(ix < n)[0],
                    torch.arange(n, n + 1, device="cuda"))
                kk2, jj2 = flatmeshgrid(
                    torch.where(ix == n)[0],
                    torch.arange(0, n + 1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:, k]

                patches[:, ix == n, 2] = torch.median(
                    patches[:, (ix == n - 1) | (ix == n - 2), 2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:, kk], corr, None,
                                                  ii, jj, kk)

            lmbda = 1e-4
            target = coords[..., p // 2, p // 2, :] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs,
                                 patches,
                                 intrinsics,
                                 target,
                                 weight,
                                 lmbda,
                                 ii,
                                 jj,
                                 kk,
                                 bounds,
                                 ep=ep,
                                 fixedp=1,
                                 structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k],
                                    kk[k])
            coords_gt, valid, _ = pops.transform(Ps,
                                                 patches_gt,
                                                 intrinsics,
                                                 ii[k],
                                                 jj[k],
                                                 kk[k],
                                                 jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:, :n], Ps[:, :n], kl))

        return traj

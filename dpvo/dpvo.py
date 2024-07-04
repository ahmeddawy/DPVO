import torch
import numpy as np
import torch.nn.functional as F

torch.set_printoptions(threshold=torch.inf)

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .net import VONet
from .utils import *
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)  # load dpvo.pt
        self.is_initialized = False
        self.enable_timing = False

        self.n = 0  # number of frames in the graph, it holds the frame index as the slam is running
        self.m = 0  # number of patches in the graph

        self.M = self.cfg.PATCHES_PER_FRAME  # the deafult config is 96 patch per frame
        self.N = self.cfg.BUFFER_SIZE  # the buffer size is 2048

        self.ht = ht  # image height
        self.wd = wd  # image width

        DIM = self.DIM  # 384 ->  comes from VONet()
        RES = self.RES  # 4   ->  comes from VONet()

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        self.image_ = torch.zeros(self.ht,
                                  self.wd,
                                  3,
                                  dtype=torch.uint8,
                                  device="cpu")

        # torch.Size([2048]), it is the size of the buffer
        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        # torch.Size([2048,7]), BUFFER_SIZE x 7
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")

        # self.P is the patch size = 3
        # torch.Size([2048, 96, 3, 3, 3]), BUFFER_SIZE x PATCHES_PER_FRAME x 3 x 3 x 3
        self.patches_ = torch.zeros(self.N,
                                    self.M,
                                    3,
                                    self.P,
                                    self.P,
                                    dtype=torch.float,
                                    device="cuda")

        # torch.Size([2048, 4]), BUFFER_SIZE x 4
        self.intrinsics_ = torch.zeros(self.N,
                                       4,
                                       dtype=torch.float,
                                       device="cuda")

        # torch.Size([196608, 3]), (BUFFER_SIZE * PATCHES_PER_FRAME) x 3
        self.points_ = torch.zeros(self.N * self.M,
                                   3,
                                   dtype=torch.float,
                                   device="cuda")

        # torch.Size([2048, 96, 3]), BUFFER_SIZE x PATCHES_PER_FRAME x 3
        self.colors_ = torch.zeros(self.N,
                                   self.M,
                                   3,
                                   dtype=torch.uint8,
                                   device="cuda")

        # torch.Size([2048, 96]), BUFFER_SIZE x PATCHES_PER_FRAME
        self.index_ = torch.zeros(self.N,
                                  self.M,
                                  dtype=torch.long,
                                  device="cuda")

        # torch.Size([2048]), BUFFER_SIZE
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        # Internsics map Size([32, 96, 384]) , to hold the intensics maps(384) of the patches(96) throughout the mem(32)
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)

        # feature map Size([32, 96, 128, 3, 3]), to hold the feature maps(128x3x3) of the patches(96) throughout the mem(32)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P,
                                 **kwargs)

        ht = ht // RES
        wd = wd // RES

        # fmap1_ Size([1, 32, 128, 132, 240]) , Size([1, 32, 128, 33, 60])
        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:, 6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.viewer = None
        if viz:
            self.start_viewer()
        self.pointcloud = []

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(self.image_, self.poses_, self.points_,
                             self.colors_, intrinsics_)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N * self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64) # float64 to fix TUM timestamp error

        if self.viewer is not None:
            self.viewer.join()
        # print("poses ", poses.shape)
        # print("pointclous ", self.points_.shape)
        # print("tstamps ", tstamps.shape)
        # pointcloud_np = self.points_.cpu().numpy()

        # # Save the NumPy array to a file
        # np.save('./plot/point_cloud.npy', pointcloud_np)
        # np.save('./plot/poses.npy', poses)
        # torch.save(self.pointcloud, './plot/pointcloud_list.pt')
        print("1st timestamps ",tstamps[0])
        print("last timestamps ",tstamps[-1])

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        '''
        - Find the correlation of gmap[ii1] and  self.pyramid[0][jj1] around the coordinates coords wich exists in frame_jj
        - Find the correlation of feature maps of patches[ii1] and the feature map of frame_jj1 around coords, with window size =3
        -- coords are the projection of patches[ii1]  to frame jj1
        '''
        # Size [1,len(ii),7,7,3,3]
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1,
                             3)

        # Size [1,len(ii),7,7,3,3]
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1,
                             3)

        # return Size[1,len(ii),882]
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj,
                                                              self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics,
                                ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:, ~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        '''1- Get the global indices of the patches of frame i
           2- Prpject these patches to frame j (which is its next frame) '''

        # The patches indices of frame index i (previous frame n-1) among the indices of all patches
        kk = torch.arange(self.m - self.M, self.m, device="cuda")
        # The current frame index
        jj = self.n * torch.ones_like(kk)
        # The frame index of the corresponding patches
        ii = self.ix[kk]
        '''
        Debugging output 
        
        n = 1
        kk  tensor([ 0 .. 95])
        jj  tensor([1 .. 1])
        ii  tensor([0 .. 0] 

        n = 2
        kk  tensor([ 96 .. 191])
        jj  tensor([2 .. 2])
        ii  tensor([1 .. 1] 
        '''

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)

        # coords of projecting patches kk from frame ii to frame jj
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            '''
                1 - Find the correlation of feature maps of patches[kk] and the feature map of frame_jj,
                    with window size =3,
                    around coords, which are the projection of patches[kk] from their original frame_ii to frame_jj.
                2 - Get the context feature maps of patches[kk]
                
            '''
            # Size[1,len(ii),882]
            corr = self.corr(coords, indicies=(kk, jj))

            # Get the context feature maps of patches[kk]
            # Size[1,96,384]
            ctx = self.imap[:, kk % (self.M * self.mem)]

            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses),
                             self.patches,
                             self.intrinsics,
                             ii,
                             jj,
                             kk,
                             beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k - 1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k - 1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n - 1):
                self.tstamps_[i] = self.tstamps_[i + 1]
                self.colors_[i] = self.colors_[i + 1]
                self.poses_[i] = self.poses_[i + 1]
                self.patches_[i] = self.patches_[i + 1]
                self.intrinsics_[i] = self.intrinsics_[i + 1]

                self.imap_[i % self.mem] = self.imap_[(i + 1) % self.mem]
                self.gmap_[i % self.mem] = self.gmap_[(i + 1) % self.mem]
                self.fmap1_[0, i % self.mem] = self.fmap1_[0,
                                                           (i + 1) % self.mem]
                self.fmap2_[0, i % self.mem] = self.fmap2_[0,
                                                           (i + 1) % self.mem]

            self.n -= 1
            self.m -= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[..., self.P // 2, self.P // 2] + delta.float()

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics, target,
                          weight, lmbda, self.ii, self.jj, self.kk, t0, self.n,
                          2)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses),
                                      self.patches[:, :self.m],
                                      self.intrinsics, self.ix[:self.m])
            points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(
                -1, 3)

            self.points_[:len(points)] = points[:]
            self.pointcloud.append(points[:])
            # print("----------------------------------------------------- ")

    def __edges_all(self):
        return flatmeshgrid(torch.arange(0, self.m, device="cuda"),
                            torch.arange(0, self.n, device="cuda"),
                            indexing='ij')

    def __edges_forw(self):
        # the configured lifetime of patches, which determines how many frames a patch should be tracked.
        r = self.cfg.PATCH_LIFETIME

        #These variables define the range of temporal indices for patches
        # This is the start index based on the current frame self.n, patch lifetime r, and a multiplier (number of patches per frame).
        t0 = self.M * max((self.n - r), 0)
        # This is the end index for the forward edges.
        t1 = self.M * max((self.n - 1), 0)

        # returns the connection between the patch index and frame index
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
                            torch.arange(self.n - 1, self.n, device="cuda"),
                            indexing='ij')

    def __edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
                            torch.arange(max(self.n - r, 0),
                                         self.n,
                                         device="cuda"),
                            indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        if self.viewer is not None:
            self.viewer.update_image(image)
            self.viewer.loop()
        # Batch the first two dimensions of the image tensor to be [1,1,3,h,w], then Normalize to the range [-0.5, 0.5]
        image = 2 * (image[None, None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            # 1- feature map of the frame Size(1, 1, 128, 132, 240)
            # 2- corresponding patches of the feature map Size([1, 96, 128, 3, 3])
            # 3- corresponding patches of the context map Size([1, 96, 384, 1, 1])
            # 4- patches of the image centered around coords with added disparity Size([1, 96, 3, 3, 3])
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    gradient_bias=self.cfg.GRADIENT_BIAS,
                    return_color=True)

        ### update state attributes ###
        print("tstamp ",tstamp)
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        # Stor the frame index for the corresponding patches
        self.index_[self.n + 1] = self.n + 1  # size(2084x96)
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n - 1])
                P2 = SE3(self.poses_[self.n - 2])

                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n - 1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        #Size([1, 96, 3, 3, 3])
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])

        if self.is_initialized:
            # update the depth by taking the median of depth across the current frame and previous 3 frames
            s = torch.median(self.patches_[self.n - 3:self.n, :, 2])
            patches[:, :, 2] = s
        self.patches_[self.n] = patches

        ### update network attributes ###
        # The memorey of the system is set in self.mem = 32
        # Thus these tensors will hold the values for the last 32 frame

        # save the context maps of the patches
        self.imap_[self.n % self.mem] = imap.squeeze()
        # save the feature maps of the patches
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        # save the feature map of the frame
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        # save the feature/4 map of the frame
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1  # increase the number of frames in the graph by 1
        self.m += self.M  # increase the number of patches in the graph by 96

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

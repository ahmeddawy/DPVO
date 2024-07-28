import torch
import numpy as np
import torch.nn.functional as F
import cv2

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


def draw_rectangle(image, top_left, patch_size, color=(0, 0, 0)):
    x, y = top_left
    image[y:y + patch_size[0], x:x + patch_size[1], :] = color
    return image


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        torch.cuda.empty_cache()
        self.cfg = cfg

        self.human_segmentor = None
        self.depth_anything = None

        if self.cfg.EXCLUDE_HUMAN:
            self.human_segmentor = YOLO('yolov8x-seg.pt')

        if self.cfg.USE_DEPTH:

            DEVICE = 'cuda' if torch.cuda.is_available(
            ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
            # DEVICE = 'cpu'
            depth_model_configs = {
                'vitl': {
                    'encoder': 'vitl',
                    'features': 256,
                    'out_channels': [256, 512, 1024, 1024]
                }
            }

            encoder = 'vitl'  # or 'vits', 'vitb'
            dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
            max_depth = 20  # 20 for indoor model, 80 for outdoor model

            self.depth_anything = DepthAnythingV2(
                **{
                    **depth_model_configs[encoder], 'max_depth': max_depth
                })
            self.depth_anything.load_state_dict(
                torch.load(f'depth_anything_v2_metric_{dataset}_{encoder}.pth',
                           map_location='cpu'))
            self.depth_anything.to(DEVICE).eval()

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
        if self.cfg.USE_DEPTH:
            image_ = torch.zeros(self.ht,
                                 self.wd * 2,
                                 3,
                                 dtype=torch.uint8,
                                 device="cpu")
        else:
            image_ = torch.zeros(self.ht,
                                 self.wd,
                                 3,
                                 dtype=torch.uint8,
                                 device="cpu")

        self.viewer = Viewer(image_, self.poses_, self.points_, self.colors_,
                             intrinsics_)

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
        stack = [t]
        result = {}

        while stack:
            current = stack[-1]
            if current in self.traj:
                result[current] = SE3(self.traj[current])
                stack.pop()
            else:
                t0, dP = self.delta[current]
                if t0 in result:
                    result[current] = dP * result[t0]
                    stack.pop()
                else:
                    stack.append(t0)

        return result[t]

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]
        #print("self.traj ", len(self.traj))
        #print("self.counter ", self.counter)

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(
            self.tlist, dtype=np.float64)  # float64 to fix TUM timestamp error

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    # def get_pose(self, t):
    #     if t in self.traj:
    #         return SE3(self.traj[t])

    #     t0, dP = self.delta[t]
    #     return dP * self.get_pose(t0)

    # def terminate(self):
    #     """ interpolate missing poses """
    #     self.traj = {}
    #     for i in range(self.n):
    #         self.traj[self.tstamps_[i].item()] = self.poses_[i]

    #     tlist = []
    #     for i in range(self.n):
    #         # print("self.tstamps_ ", self.tstamps_[i])
    #         # print("self.traj ", self.traj[self.tstamps_[i].item()])
    #         tlist.append(self.tstamps_[i].item())
        
        
        
        
    #     poses = [SE3(self.traj[t]) for t in tlist]
    #     poses = lietorch.stack(poses, dim=0)
    #     poses = poses.inv().data.cpu().numpy()
        
        
        
    #     tstamps = np.array(
    #         tlist, dtype=np.float64)  # float64 to fix TUM timestamp error

    #     if self.viewer is not None:
    #         self.viewer.join()

    #     return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        # self.kk holds the indices of the patches , self.jj holds the frame index to which patches kk was projected
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)

        # ii1 holds the patches indices, j11 holds the frame index
        '''
        - Find the correlation of gmap[ii1] and  self.pyramid[0][jj1] around the coordinates coords wich exists in frame_jj
        - Find the correlation of feature maps of patches[ii1] and the feature map of frame_jj1 around coords, with window size =3
        -- coords are the projection of patches[ii1]  to frame jj1
        '''
        """
            - self.gmap  [1, 32 * num_patches_per_frame, 128, 3, 3] -> [batch_size, total number of patches in the memeroy window, channels , patch width, patch height]
            - self.pyramid[0]  [1, 32 , 128, 132, 240] -> [batch_size, total number of frames in the memeroy window, channels , width, height]

            --- self.gmap[0][ii1] is [ii1.shape, 128, 3, 3]
            --- self.pyramid[0][0][jj1] is [jj1.shape, 128, 132, 240]
        
        """
        # Size [1,len(ii),7,7,3,3] -> Size[1,Num of Patches to compute coo , 7,7,3,3]

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
        # ii is the patch index
        # jj the frame index to which patch ii is connected

        self.jj = torch.cat([self.jj, jj])  # frame indices
        self.kk = torch.cat([self.kk, ii])  # patches indices
        self.ii = torch.cat([
            self.ii, self.ix[ii]
        ])  # holds the indices of the original patch locations in the frame

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
           2- Project these patches to frame j (which is its next frame) '''
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
        # [1, 96, 2, 3, 3] -> [batch_size, M(number of Patches per frame) ,2 (channels) , x coordinates of the patch(3) , y coordinates of the patch (3)]

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
            # self.imap [1,memory window x number of patches per frame, 384 ] -> contains patches context feature maps
            # ctx [1,  number of patches per frame , 384] -> contains patches_kk context feature maps
            ctx = self.imap[:, kk % (self.M * self.mem)]

            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

            # net size [batch_size , number of patches per frame,384]
            # delta size is [batch_size , number of patches per frame , 2 ]
            # weight size is [batch_size , number of patches per frame , 2 ]
            """ 1- Calculate the norm of the delta returns along dim=-1, which means calculate the norm of the returned delta features for each patch
                2- Sort the values and Get the median of all the values"""

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

        # self.cfg.KEYFRAME_INDEX = 4

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1

        # Get the motion from i -> j and from i <- j
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:  # Get the average of this motion
            # the current frame is a candidate for removal
            k = self.n - self.cfg.KEYFRAME_INDEX  # the index of the frame that is candidtae for removal (it is the in between i and j )

            #t0 and t1 are the timestamps of the frames k-1 and k, respectively.
            t0 = self.tstamps_[k - 1].item()
            t1 = self.tstamps_[k].item()

            # dP is the relative pose transformation between frames k and k-1
            dP = SE3(self.poses_[k]) * SE3(self.poses_[k - 1]).inv()
            self.delta[t1] = (
                t0, dP
            )  # store the relative pose transformation and the corresponding timestamp

            # Get the indices at which the frame index = the rquired to be removed index
            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            #This mask identifies the patches whose source frames come after the removed frame k, So correct their indices by removing the M
            self.kk[self.ii > k] -= self.M
            #This mask identifies the source frames whose come after the removed frame k, So correct their indices by removing 1
            self.ii[self.ii > k] -= 1
            #Adjusts the frame indices in self.jj to account for the removed frame, ensuring subsequent frames are correctly re-indexed.
            self.jj[self.jj > k] -= 1

            # Update all the attributes of the class accordingly
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
        # Remove frames that their indices are outdated
        # 1- self.ix[self.kk] -> Get the source frame indices of the current patches
        # 2- Compare the source frame indices with (self.n - self.cfg.REMOVAL_WINDOW)
        # 3- Get the True masks for source frames where they are < (self.n - self.cfg.REMOVAL_WINDOW)
        # --> Ex if self.n = 30 and self.cfg.REMOVAL_WINDOW = 22
        # -- any frame index < 8 will be removed with its patches
        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            '''
               In this function we do the exact same as before the init, except we pass the full graph (self.ii , self.jj, self,kk) 
               instead of passing the patches of the prev frame only.
               Thus all the shapes will represent the shape of the graph (ii.shape)
            '''
            #-> Size[batch_size, ii.shape , num_channels , patch_width , patch_height] = [1,ii.shape , 2, 3 ,3 ]
            # contains the reprojected coordinates of patches (self.kk) from their source frames (self.ii) to their target frames (self.jj)
            coords = self.reproject()
            with autocast(enabled=True):
                #-> Size[batch_size, ii.shape , num_channels] = [1,ii.shape , 882]
                corr = self.corr(coords)
                #-> Size [batch_size, ii.shape ,num_channels ] = [1,ii.shape , 384]
                ctx = self.imap[:, self.kk % (self.M * self.mem)]

                # self.net -> Size [batch_size , ii.shape,384]
                # delta -> Size [batch_size , ii.shape , 2 ]
                # weight -> Size [batch_size , ii.shape , 2 ]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()

            # This operation -coords[..., self.P // 2, self.P // 2]- reduces the last two dimensions (patch grid) to a single coordinate,
            # giving the central coordinate of each patch.
            target = coords[..., self.P // 2, self.P // 2] + delta.float(
            )  # -> Size [batch_size , ii.shape , 2]

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
        '''constructs edges connecting patches from the previous frames to 
           the current frame within the specified distance r.'''
        # the configured lifetime of patches, which determines how many frames a patch should be tracked.
        r = self.cfg.PATCH_LIFETIME

        #These variables define the range of temporal indices for patches

        # This is the patches start index based on the current frame self.n, patch lifetime r, and a multiplier (number of patches per frame).
        t0 = self.M * max((self.n - r), 0)
        # This is the patches end index for the forward edges.
        t1 = self.M * max((self.n - 1), 0)

        # returns the connection between the patch index and frame index
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
                            torch.arange(self.n - 1, self.n, device="cuda"),
                            indexing='ij')

    def __edges_back(self):
        '''constructs edges connecting patches from the current frame
          to the previous frames within the specified distance r.'''
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)

        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
                            torch.arange(max(self.n - r, 0),
                                         self.n,
                                         device="cuda"),
                            indexing='ij')

    def __call__(self,
                 tstamp,
                 image,
                 intrinsics,
                 human_masks=None,
                 disparity=None):
        """ track new frame """

        image_np = image.detach().clone()
        image_np = image_np.permute(1, 2, 0)
        image_np = image_np.cpu().numpy()

        viewer_image_np = image_np.copy()

        if self.cfg.EXCLUDE_HUMAN:
            human_masks = get_human_masks(image_np, self.human_segmentor)

        if self.cfg.USE_DEPTH:
            depth_np, disparity_np = get_disparity(image_np,
                                                   self.depth_anything)
            disparity = torch.from_numpy(disparity_np)
            disparity = disparity[None, None].to('cuda')
            # print("depth_np ",depth_np.shape)

        # Batch the first two dimensions of the image tensor to be [1,1,3,h,w], then Normalize to the range [-0.5, 0.5]
        image = 2 * (image[None, None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            # 1- feature map of the frame Size(1, 1, 128, 132, 240)
            # 2- corresponding patches of the feature map Size([1, 96, 128, 3, 3])
            # 3- corresponding patches of the context map Size([1, 96, 384, 1, 1])
            # 4- patches of the image centered around coords with added disparity Size([1, 96, 3, 3, 3])
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,disps=disparity,
                    gradient_bias=self.cfg.GRADIENT_BIAS,
                    return_color=True, human_masks=human_masks)

        if self.cfg.EXCLUDE_HUMAN:
            patches_np = patches.squeeze().cpu().numpy()
            x_coords = patches_np[:, 0, :, :].reshape(-1, 3 * 3)
            y_coords = patches_np[:, 1, :, :].reshape(-1, 3 * 3)
            x_min = (x_coords.min(axis=1)) * 4
            x_max = (x_coords.max(axis=1)) * 4
            y_min = (y_coords.min(axis=1)) * 4
            y_max = (y_coords.max(axis=1)) * 4

            # Draw rectangles on the image
            for i in range(len(x_min)):
                start_point = (int(x_min[i]), int(y_min[i]))  # Top-left corner
                end_point = (int(x_max[i]), int(y_max[i])
                             )  # Bottom-right corner

                rect_points = np.array(
                    [
                        [start_point[0], start_point[1]],  # Top-left
                        [end_point[0], start_point[1]],  # Top-right
                        [end_point[0], end_point[1]],  # Bottom-right
                        [start_point[0], end_point[1]]  # Bottom-left
                    ],
                    dtype=np.int32)
                color = (0, 0, 0)
                cv2.fillPoly(viewer_image_np, [rect_points], color)

            if human_masks is not None:
                # Populate the mask overlay with the mask pixels
                human_mask_overlay = np.zeros(viewer_image_np.shape[:2],
                                              dtype=np.uint8)
                for pixel in human_masks:
                    human_mask_overlay[pixel[0], pixel[1]] = 255

                # Create a colored version of the mask overlay
                colored_mask_overlay = np.zeros_like(viewer_image_np)
                colored_mask_overlay[human_mask_overlay == 255] = [
                    0, 255, 0
                ]  # Green color for the mask

                # Overlay the viewer image and the colored human mask
                viewer_image_np = cv2.addWeighted(viewer_image_np, 1,
                                                  colored_mask_overlay, 0.5, 0)

        if self.cfg.USE_DEPTH:
            # cv2.imwrite("./depth.jpg",)
            # Stack images horizontally
            # viewer_image_np = np.hstack((viewer_image_np, depth_np*255/depth_np.max()))
            depth_np = depth_np * 255 / depth_np.max()
            depth_np = depth_np.astype(np.uint8)

            viewer_image_np = np.hstack((viewer_image_np, depth_np))

        viewer_image = torch.from_numpy(viewer_image_np)
        viewer_image = viewer_image.permute(2, 0, 1)

        if self.viewer is not None:
            self.viewer.update_image(viewer_image)
            self.viewer.loop()

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
            if self.motion_probe() < 0.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1  # increase the number of frames in the graph by 1
        self.m += self.M  # increase the number of patches in the graph by 96

        # print("n ", self.n)
        # print("self.ii b", self.ii.shape)
        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())
        # print("self.ii a", self.ii.shape)

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            # for i in range (4):
            self.update()
            self.keyframe()

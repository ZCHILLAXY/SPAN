import torch
import torch.nn as nn
from config import cfg

class PointPillarScatter(nn.Module):
    def __init__(self,
                 num_bev_features=64,
                 template=False):
        super().__init__()

        self.num_bev_features = num_bev_features
        if template:
            self.nx = cfg.TEMPLATE_INPUT_WIDTH
            self.ny = cfg.TEMPLATE_INPUT_HEIGHT
            self.nz = cfg.TEMPLATE_INPUT_DEPTH
        else:
            self.nx = cfg.SCENE_INPUT_WIDTH
            self.ny = cfg.SCENE_INPUT_HEIGHT
            self.nz = cfg.SCENE_INPUT_DEPTH
        assert self.nz == 1

    def forward(self, x_0, x_1, batchsize):
        x_0 = x_0.view(-1, 64)
        pillar_features, coords = x_0, x_1
        batch_spatial_features = []
        batch_size = batchsize
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        return batch_spatial_features

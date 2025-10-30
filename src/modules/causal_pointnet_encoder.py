# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalPointNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        layers = []
        for i in range(num_pre_layers):
            layers.append(nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.pre_mlps = nn.Sequential(*layers)
        layers = []
        for i in range(num_layers - num_pre_layers):
            layers.append(nn.Linear(hidden_dim * 2 if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.mlps = nn.Sequential(*layers)

        if out_channels is not None:
            layers = []
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_channels))
            self.out_mlps = nn.Sequential(*layers)
        else:
            self.out_mlps = None

    def forward(self, polylines, return_cache=False):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape
        polylines = rearrange(polylines, "b a t d -> (b a) t d")
        inference_cache = {}
        # pre-mlp
        polylines = rearrange(polylines, "ba t d -> (ba t) d")
        polylines_feature = self.pre_mlps(polylines)
        polylines_feature = rearrange(
            polylines_feature, "(ba t) d -> ba d t", t=num_points_each_polylines
        )
        if return_cache:
            inference_cache["polylines_feature_pre_mlps"] = polylines_feature.detach().clone()

        # get Lookback global feature
        polylines_feature_padded = F.pad(
            polylines_feature,
            (num_points_each_polylines - 1, 0),
            "constant",
            0,
        )
        pooled_feature = F.max_pool1d(
            polylines_feature_padded,
            kernel_size=num_points_each_polylines,
            stride=1,
        )  # [ba, d, t]
        polylines_feature = torch.cat((polylines_feature, pooled_feature), dim=-2)

        # mlp
        polylines_feature = rearrange(polylines_feature, "ba d t -> (ba t) d")
        feature_buffers = self.mlps(polylines_feature)
        feature_buffers = rearrange(
            feature_buffers, "(ba t) d -> ba d t", t=num_points_each_polylines
        )
        if return_cache:
            inference_cache["polylines_feature_mlps"] = feature_buffers.detach().clone()

        # max-pooling
        feature_buffers_padded = F.pad(
            feature_buffers,
            (num_points_each_polylines - 1, 0),
            "constant",
            0,
        )
        feature_buffers = F.max_pool1d(
            feature_buffers_padded,
            kernel_size=num_points_each_polylines,
            stride=1,
        )

        feature_buffers = rearrange(feature_buffers, "ba d t -> ba t d")

        # out-mlp
        if self.out_mlps is not None:
            feature_buffers = rearrange(feature_buffers, "ba t d -> (ba t) d")
            feature_buffers = self.out_mlps(feature_buffers)
            feature_buffers = rearrange(
                feature_buffers, "(ba t) d -> ba t d", t=num_points_each_polylines
            )

        feature_buffers = rearrange(
            feature_buffers, "(b a) t d -> b a t d", b=batch_size, a=num_polylines
        )
        return feature_buffers, inference_cache

    def generate(self, x: torch.tensor, inference_cache: dict):
        """
        x: [b, num_agents, 1, d]
        inference_cache: dict, will be updated in place
        """
        polylines = rearrange(x, "b a 1 d -> (b a) d")
        polylines_feature = self.pre_mlps(polylines)
        polylines_feature = rearrange(polylines_feature, "ba d -> ba d 1")
        polylines_feature = torch.cat(
            [inference_cache["polylines_feature_pre_mlps"], polylines_feature], dim=-1
        )
        inference_cache["polylines_feature_pre_mlps"] = polylines_feature.clone()
        pooled_feature = polylines_feature.max(dim=-1)[0]  # [ba, d]
        polylines_feature = torch.cat(
            [
                polylines_feature[..., -1],  # [ba, d]
                pooled_feature,
            ],
            dim=-1,
        )
        feature_buffers = self.mlps(polylines_feature)
        feature_buffers = rearrange(feature_buffers, "ba d -> ba d 1")
        feature_buffers = torch.cat(
            [inference_cache["polylines_feature_mlps"], feature_buffers], dim=-1
        )
        inference_cache["polylines_feature_mlps"] = feature_buffers.clone()

        feature_buffers = feature_buffers.max(dim=-1)[0]  # [ba, d]
        if self.out_mlps is not None:
            feature_buffers = self.out_mlps(feature_buffers)

        feature_buffers = rearrange(
            feature_buffers, "(b a) d -> b a 1 d", b=x.shape[0], a=x.shape[1]
        )
        return feature_buffers


if __name__ == "__main__":
    encoder = SequentialPointNetPolylineEncoder(
        in_channels=2, hidden_dim=4, num_layers=3, num_pre_layers=1, out_channels=4
    )
    encoder.eval()
    polylines = torch.rand(16, 11, 20, 2)
    feature_buffers, inference_cache = encoder(polylines[:, :, :-1], return_cache=True)
    print(feature_buffers.shape)
    feature_buffers_gen = encoder.generate(polylines[:, :, -1:], inference_cache)
    print(feature_buffers_gen.shape)

    feature_buffers_all, _ = encoder(polylines, return_cache=False)
    print(feature_buffers_all.shape)

    assert torch.allclose(feature_buffers_all[:, :, -1:], feature_buffers_gen)

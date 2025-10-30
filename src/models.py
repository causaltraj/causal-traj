from sqlite3.dbapi2 import converters

import torch
import torch.nn as nn
from einops import rearrange

from src.modules.causal_pointnet_encoder import CausalPointNetEncoder
from src.modules.mamba2mixer_encoder import Mamba2MixerEncoder
from src.modules.spatial_relation_transformer_block import SpatialRelationTransformerBlock


class CausalTrajModel(nn.Module):
    def __init__(
        self,
        d_traj,
        agent_embedding_dim,
        causal_pointnet_config,
        mamba2mixer_config,
        num_transformer_blocks,
        transformer_block_config,
        num_spatial_relation_transformer_blocks,
        spatial_relation_transformer_block_config,
        d_agentwise_mlp,
        d_shared_head_mlp,
        num_scenes,
        num_agents,
        traj_conditioning=True,
    ):
        super().__init__()
        self.traj_conditioning = traj_conditioning
        self.team_one_query_embedding = nn.Embedding(1, agent_embedding_dim)
        self.team_two_query_embedding = nn.Embedding(1, agent_embedding_dim)
        self.ball_query_embedding = nn.Embedding(1, agent_embedding_dim)
        assert (
            mamba2mixer_config is None or causal_pointnet_config is None
        ), "Only one of mamba2mixer_config or causal_pointnet_config can be provided"
        self.mamba2mixer_config = mamba2mixer_config
        self.causal_pointnet_config = causal_pointnet_config
        if mamba2mixer_config is not None:
            self.past_encoder = Mamba2MixerEncoder(**mamba2mixer_config)
            past_encoder_d_model = mamba2mixer_config["d_model"]
        elif causal_pointnet_config is not None:
            self.past_encoder = CausalPointNetEncoder(**causal_pointnet_config)
            past_encoder_d_model = causal_pointnet_config["out_channels"]
        self.mlp_z = nn.Sequential(
            nn.Linear(
                agent_embedding_dim + past_encoder_d_model, transformer_block_config["d_model"]
            ),
            nn.ReLU(),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_block_config["d_model"],
                nhead=transformer_block_config["n_head"],
                dim_feedforward=transformer_block_config["d_ff"],
                norm_first=transformer_block_config["use_pre_norm"],
                dropout=transformer_block_config["dropout"],
                batch_first=True,
            ),
            num_layers=num_transformer_blocks,
        )
        self.spatial_relation_transformer_blocks = nn.ModuleList(
            [
                SpatialRelationTransformerBlock(**spatial_relation_transformer_block_config)
                for _ in range(num_spatial_relation_transformer_blocks)
            ]
        )
        self.post_act = nn.ReLU()

        # use num agentwise_mlp layers to set
        agentwise_mlp_layers = [
            nn.Linear(
                (
                    spatial_relation_transformer_block_config["d_model"]
                    if not traj_conditioning
                    else spatial_relation_transformer_block_config["d_model"] + d_traj
                ),
                d_agentwise_mlp[0],
            ),
            nn.ReLU(),
        ]
        for i in range(1, len(d_agentwise_mlp)):
            agentwise_mlp_layers.extend(
                [
                    nn.Linear(d_agentwise_mlp[i - 1], d_agentwise_mlp[i]),
                    nn.ReLU(),
                ]
            )
        self.agentwise_mlp = nn.Sequential(*agentwise_mlp_layers)
        self.shared_head = nn.Sequential(
            nn.Linear(
                num_agents * d_agentwise_mlp[-1],
                d_shared_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(
                d_shared_head_mlp,
                d_shared_head_mlp,
            ),
            nn.ReLU(),
            nn.Linear(
                d_shared_head_mlp,
                num_scenes + (num_scenes * num_agents * 3) + (num_scenes * num_agents * 2),
            ),
        )
        self.num_scenes = num_scenes
        self.num_agents = num_agents
        self.team_size = num_agents // 2

    def agent_query_embedding(self, index):
        """
        Distinguish between team one, team two and ball. High level PE
        One team is at index 0-5
        Another team is at index 5-10
        Ball is at index 10
        """
        team_one_query = self.team_one_query_embedding(index)
        team_two_query = self.team_two_query_embedding(index)
        ball_query = self.ball_query_embedding(index)
        agent_query = torch.cat(
            [
                team_one_query.repeat(self.team_size, 1),
                team_two_query.repeat(self.team_size, 1),
                ball_query,
            ],
            dim=0,
        )
        return agent_query  # [A, D]

    def forward(self, x_traj: torch.tensor, return_cache=False):
        inference_cache = {}
        # history encoder
        past_traj = rearrange(x_traj, "b t a d -> b a t d")
        z_past_traj, past_encoder_cache = self.past_encoder(past_traj, return_cache=return_cache)
        if return_cache:
            inference_cache["past_encoder_cache"] = past_encoder_cache
        z_past_traj = rearrange(z_past_traj, "b a t d -> (b t) a d")

        # agent query embedding
        agent_query = rearrange(
            self.agent_query_embedding(torch.arange(1).to(past_traj.device)), "a d -> 1 a d"
        ).repeat(z_past_traj.shape[0], 1, 1)
        z = torch.cat([z_past_traj, agent_query], dim=-1)
        z = self.mlp_z(z)

        # inter-agent relation encoder
        z = self.transformer_encoder(z)
        z = rearrange(z, "(b t) a d -> b t a d", b=past_traj.shape[0])

        for block in self.spatial_relation_transformer_blocks:
            z = block(x_traj, z)  # [b, t, a, d]
        z = self.post_act(z)

        # scene concatenation and prediction head
        if self.traj_conditioning:
            z = torch.cat([z, x_traj], dim=-1)

        z = self.agentwise_mlp(z)
        z = rearrange(z, "b t a d -> b t (a d)")

        out = self.shared_head(z)  # [b, t, d]

        cls_out = out[:, :, : self.num_scenes]  # [b, t, num_scenes]
        cov_out = out[
            :, :, self.num_scenes : self.num_scenes + (self.num_scenes * self.num_agents * 3)
        ]  # [b, t, num_scenes * num_agents * 2]
        cov_out = rearrange(
            cov_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        reg_out = out[:, :, self.num_scenes + (self.num_scenes * self.num_agents * 3) :]
        reg_out = rearrange(
            reg_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        return reg_out, cls_out, cov_out, inference_cache

    def generate(self, x: torch.tensor, inference_cache: dict):
        """
        x: [b, 1, num_agents, d]
        inference_cache: dict, will be updated in place
        """
        past_traj = rearrange(x, "b t a d -> b a t d")
        if self.mamba2mixer_config:
            z_past_traj, _ = self.past_encoder(
                past_traj, inference_params=inference_cache["past_encoder_cache"]
            )
        else:
            z_past_traj = self.past_encoder.generate(
                past_traj, inference_cache["past_encoder_cache"]
            )  # z_past_traj: [b, 1, a, d]
        z_past_traj = rearrange(z_past_traj, "b a t d -> (b t) a d")
        agent_query = rearrange(
            self.agent_query_embedding(torch.arange(1).to(past_traj.device)), "a d -> 1 a d"
        ).repeat(z_past_traj.shape[0], 1, 1)
        z = torch.cat([z_past_traj, agent_query], dim=-1)
        z = self.mlp_z(z)

        z = self.transformer_encoder(z)
        z = rearrange(z, "(b t) a d -> b t a d", b=past_traj.shape[0])
        for block in self.spatial_relation_transformer_blocks:
            z = block(x, z)  # [b, t, a, d]
        z = self.post_act(z)

        if self.traj_conditioning:
            z = torch.cat([z, x], dim=-1)
        z = self.agentwise_mlp(z)
        z = rearrange(z, "b t a d -> b t (a d)")
        out = self.shared_head(z)  # [b, t, d]
        cls_out = out[:, :, : self.num_scenes]  # [b, t, num_scenes]
        cov_out = out[
            :, :, self.num_scenes : self.num_scenes + (self.num_scenes * self.num_agents * 3)
        ]  # [b, t, num_scenes * num_agents * 3]
        cov_out = rearrange(
            cov_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        reg_out = out[:, :, self.num_scenes + (self.num_scenes * self.num_agents * 3) :]
        reg_out = rearrange(
            reg_out, "b t (k a d) -> b t k a d", k=self.num_scenes, a=self.num_agents
        )
        return reg_out, cls_out, cov_out


if __name__ == "__main__":
    # model = CausalTrajModel(
    #     d_traj=4,
    #     agent_embedding_dim=64,
    #     causal_pointnet_config={
    #         "in_channels": 4,
    #         "hidden_dim": 64,
    #         "num_layers": 3,
    #         "num_pre_layers": 1,
    #         "out_channels": 64,
    #     },
    #     mamba2mixer_config=None,
    #     num_transformer_blocks=4,
    #     transformer_block_config={
    #         "d_model": 128,
    #         "n_head": 8,
    #         "d_ff": 512,
    #         "use_pre_norm": True,
    #         "dropout": 0.1,
    #     },
    #     num_spatial_relation_transformer_blocks=4,
    #     spatial_relation_transformer_block_config={
    #         "d_model": 128,
    #         "d_mesh": 260,
    #         "n_head": 8,
    #         "d_ff": 256,
    #         "dropout": 0.1,
    #         "use_pffn": True,
    #         "ln_before_ffn": True,
    #     },
    #     d_agentwise_mlp=[64],
    #     d_shared_head_mlp=768,
    #     num_scenes=8,
    #     num_agents=11,
    #     traj_conditioning=True,
    # ).to("cuda")

    model = CausalTrajModel(
        d_traj=4,
        agent_embedding_dim=64,
        causal_pointnet_config=None,
        mamba2mixer_config={
            "in_channels": 4,
            "n_layer": 3,
            "d_model": 64,
            "d_intermediate": 0,
            "ssm_cfg": {
                "layer": "Mamba2",
                "d_state": 128,
                "d_conv": 4,
                "expand": 4,
                "headdim": 16,
                "ngroups": 1,
                "chunk_size": 32,
                "bias": False,
                "conv_bias": False,
            },
            "rms_norm": True,
            "fused_add_norm": True,
            "residual_in_fp32": True,
        },
        num_transformer_blocks=4,
        transformer_block_config={
            "d_model": 128,
            "n_head": 8,
            "d_ff": 512,
            "use_pre_norm": True,
            "dropout": 0.1,
        },
        num_spatial_relation_transformer_blocks=4,
        spatial_relation_transformer_block_config={
            "d_model": 128,
            "d_mesh": 260,
            "n_head": 8,
            "d_ff": 256,
            "dropout": 0.1,
            "use_pffn": True,
            "ln_before_ffn": True,
        },
        d_agentwise_mlp=[64],
        d_shared_head_mlp=768,
        num_scenes=8,
        num_agents=11,
        traj_conditioning=True,
    ).to("cuda")

    model.eval()
    x = torch.rand(10, 20, 11, 4).to("cuda")
    # with torch.no_grad():
    reg_out, cls_out, shrink_out, inference_cache = model(x[:, :-1], return_cache=True)
    print(reg_out.shape)
    print(cls_out.shape)
    print(shrink_out.shape)

    reg_out_gen, cls_out_gen, shrink_out_gen = model.generate(x[:, -1:], inference_cache)
    print(reg_out_gen.shape)
    print(cls_out_gen.shape)
    print(shrink_out_gen.shape)

    reg_out_all, cls_out_all, shrink_out_all, _ = model(x, return_cache=False)
    print(reg_out_all.shape)
    print(cls_out_all.shape)
    print(shrink_out_all.shape)

    assert torch.allclose(reg_out_all[:, -1:], reg_out_gen, atol=1e-5)
    assert torch.allclose(cls_out_all[:, -1:], cls_out_gen, atol=1e-5)
    assert torch.allclose(shrink_out_all[:, -1:], shrink_out_gen, atol=1e-5)

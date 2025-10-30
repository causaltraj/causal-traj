import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w3(x) * (self.activation(self.w1(x)))
        x = self.w2(x)
        return x


class SpatialRelationAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_mesh,
        n_head,
        use_traj_diff=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mesh = d_mesh
        self.n_head = n_head
        self.key_mesh_feedforward = nn.Linear(d_mesh, d_model)
        self.value_mesh_feedforward = nn.Linear(d_mesh, d_model)
        self.query_feedforward = nn.Linear(d_model, d_model)
        self.use_traj_diff = use_traj_diff

    def forward(
        self,
        x_traj,
        x_embeddings,
    ):
        """
        x_traj: [b, t, a, d]
        x_embeddings: [b, t, a, d]
        """
        num_agents = x_traj.shape[-2]
        diff_mesh = x_traj.unsqueeze(-3) - x_traj.unsqueeze(-2)  # [b, t, a, a, d]
        q_agent_embeddings = rearrange(x_embeddings, "b t a d -> b t a 1 d").repeat(
            1, 1, 1, num_agents, 1
        )  # [b, t, a, a, d]
        k_agent_embeddings = rearrange(x_embeddings, "b t a d -> b t 1 a d").repeat(
            1, 1, num_agents, 1, 1
        )  # [b, t, a, a, d]
        if self.use_traj_diff:
            x_mesh = torch.cat(
                [diff_mesh, q_agent_embeddings, k_agent_embeddings], dim=-1
            )  # [b, t, a, a, d]
        else:
            x_mesh = torch.cat([q_agent_embeddings, k_agent_embeddings], dim=-1)  # [b, t, a, a, d]
        x_query = self.query_feedforward(x_embeddings)  # [b, t, a, d]
        x_key = self.key_mesh_feedforward(x_mesh)  # [b, t, a, a, d]
        x_value = self.value_mesh_feedforward(x_mesh)  # [b, t, a, a, d]
        x_query = rearrange(x_query, "b t a (h d) -> b t a h 1 d", h=self.n_head)
        x_key = rearrange(x_key, "b t a r (h d) -> b t a h r d", h=self.n_head)
        x_value = rearrange(x_value, "b t a r (h d) -> b t a h r d", h=self.n_head)
        x = F.scaled_dot_product_attention(x_query, x_key, x_value, None)  # [b, t, a, h, 1, d]
        x = rearrange(x, "b t a h 1 d -> b t a (h d)")
        return x


class SpatialRelationTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_mesh,
        n_head,
        d_ff,
        dropout=0.1,
        use_traj_diff=True,
        use_pffn=True,
        ln_before_ffn=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mesh = d_mesh
        self.n_head = n_head
        self.spatial_relation_attention = SpatialRelationAttention(
            d_model, d_mesh, n_head, use_traj_diff
        )

        if use_pffn:
            self.ffn = PositionwiseFFN(d_model, d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln_before_ffn = ln_before_ffn

    def forward(self, x_traj, x_embeddings):
        spatial_relation_attention_out = self.spatial_relation_attention(
            x_traj, self.ln(x_embeddings)
        )
        x_embeddings = x_embeddings + self.dropout(spatial_relation_attention_out)
        if self.ln_before_ffn:
            ffn_out = self.ffn(self.ln(x_embeddings))
        else:
            ffn_out = self.ffn(x_embeddings)
        x_embeddings = x_embeddings + self.dropout(ffn_out)
        return x_embeddings


if __name__ == "__main__":
    block = SpatialRelationTransformerBlock(64, 64 + 64 + 2, 4, 128)
    x_traj = torch.rand(64, 50, 11, 2)
    x_embeddings = torch.rand(64, 50, 11, 64)
    x = block(x_traj, x_embeddings)
    print(x.shape)
